"""NEUROSCAN — Adversarial string discovery engine.

Uses nanoGCG for Greedy Coordinate Gradient attacks to find
adversarial suffixes that elicit target outputs from LLMs.

Loads the raw HuggingFace model (not TransformerLens) because nanogcg
needs a PreTrainedModel with .get_input_embeddings() and .forward().
Runs the GCG loop step-by-step (not via nanogcg.run()) so we get
per-step progress streaming and stop support.
"""

import asyncio
import copy
import logging
import threading

logger = logging.getLogger("neuroscan.adversarial")

# Map from our model registry names to HuggingFace model IDs
_HF_MODEL_MAP = {
    "gpt2-small": "gpt2",
    "pythia-70m": "EleutherAI/pythia-70m-deduped",
    "gemma-2-2b": "google/gemma-2-2b",
}


class AdversarialEngine:
    """Manages GCG adversarial attacks with real-time progress streaming."""

    def __init__(self, model_mgr):
        self._model_mgr = model_mgr
        self._running = False
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially
        self._thread = None
        self._gcg = None  # nanogcg.gcg.GCG instance (for stop support)
        self._best_loss = float('inf')
        self._best_suffix = ""
        self._current_step = 0
        self._total_steps = 0

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "paused": self._paused,
            "available": True,
            "step": self._current_step,
            "total_steps": self._total_steps,
            "best_loss": self._best_loss if self._best_loss < float('inf') else None,
            "best_suffix": self._best_suffix,
        }

    def start_attack(self, config: dict, progress_callback, loop):
        """Start a GCG attack in a background thread."""
        if self._running:
            logger.warning("Attack already running")
            return

        self._running = True
        self._paused = False
        self._pause_event.set()
        self._best_loss = float('inf')
        self._best_suffix = ""
        self._current_step = 0
        self._total_steps = config.get("num_steps", 250)

        def _run():
            try:
                self._run_gcg(config, progress_callback, loop)
            except Exception as e:
                logger.error(f"GCG attack error: {e}", exc_info=True)
                asyncio.run_coroutine_threadsafe(
                    progress_callback({"step": self._current_step, "error": str(e), "complete": True}),
                    loop
                )
            finally:
                self._running = False
                self._paused = False
                self._gcg = None

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def pause_attack(self):
        """Pause the GCG loop after current step."""
        if self._running and not self._paused:
            self._paused = True
            self._pause_event.clear()
            logger.info("GCG attack paused")

    def resume_attack(self):
        """Resume a paused GCG loop."""
        if self._running and self._paused:
            self._paused = False
            self._pause_event.set()
            logger.info("GCG attack resumed")

    def stop_attack(self):
        """Signal the GCG loop to stop after current step."""
        self._running = False
        self._paused = False
        self._pause_event.set()  # unblock if paused so thread can exit
        if self._gcg is not None:
            self._gcg.stop_flag = True

    def _run_gcg(self, config, progress_callback, loop):
        """Execute the GCG optimization loop with per-step progress."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from nanogcg.gcg import GCG, GCGConfig, GCGResult, sample_ids_from_grad, filter_ids
        from accelerate.utils import find_executable_batch_size

        model_name = self._model_mgr.current_model
        if not model_name:
            raise RuntimeError("No model loaded")

        hf_model_id = _HF_MODEL_MAP.get(model_name)
        if not hf_model_id:
            raise RuntimeError(f"No HuggingFace mapping for model: {model_name}")

        target = config["target"]
        num_steps = config.get("num_steps", 250)
        search_width = config.get("batch_size", 64)
        top_k = config.get("top_k", 256)
        suffix_length = config.get("suffix_length", 20)

        # ── Load the raw HF model ──────────────────────────────────
        logger.info(f"Loading HF model {hf_model_id} for GCG attack")
        asyncio.run_coroutine_threadsafe(
            progress_callback({"step": 0, "message": f"Loading {hf_model_id}...", "loading": True}),
            loop,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            torch_dtype=torch.float16,
            device_map=device,
        )
        hf_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Configure GCG ──────────────────────────────────────────
        init_suffix = " ".join(["x"] * suffix_length)
        gcg_config = GCGConfig(
            num_steps=num_steps,
            optim_str_init=init_suffix,
            search_width=search_width,
            topk=top_k,
            seed=42,
            verbosity="WARNING",  # suppress nanogcg's own logging
        )

        # The message is just the adversarial suffix placeholder.
        # For base models (GPT-2, Pythia), nanogcg auto-sets a passthrough
        # chat template so the content is passed directly.
        messages = [{"role": "user", "content": "{optim_str}"}]

        logger.info(f"Starting GCG: target='{target}', steps={num_steps}, "
                    f"search_width={search_width}, suffix_len={suffix_length}")

        # ── Initialize GCG manually (instead of nanogcg.run) ──────
        gcg = GCG(hf_model, tokenizer, gcg_config)
        self._gcg = gcg

        # Replicate GCG.run() setup
        msgs = copy.deepcopy(messages)
        if not any("{optim_str}" in d["content"] for d in msgs):
            msgs[-1]["content"] = msgs[-1]["content"] + "{optim_str}"

        template = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")

        before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(hf_model.device, torch.int64)
        after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(hf_model.device, torch.int64)
        target_ids = tokenizer([target], add_special_tokens=False, return_tensors="pt")["input_ids"].to(hf_model.device, torch.int64)

        embedding_layer = gcg.embedding_layer
        before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]

        if gcg_config.use_prefix_cache and before_ids.shape[1] > 0:
            with torch.no_grad():
                output = hf_model(inputs_embeds=before_embeds, use_cache=True)
                gcg.prefix_cache = output.past_key_values

        gcg.target_ids = target_ids
        gcg.before_embeds = before_embeds
        gcg.after_embeds = after_embeds
        gcg.target_embeds = target_embeds

        buffer = gcg.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []

        # ── Per-step GCG loop ──────────────────────────────────────
        for step in range(num_steps):
            # Wait here if paused (blocks until resume or stop)
            self._pause_event.wait()
            if not self._running:
                logger.info("GCG stopped by user")
                break

            optim_ids_onehot_grad = gcg.compute_token_gradient(optim_ids)

            with torch.no_grad():
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    gcg_config.search_width,
                    gcg_config.topk,
                    gcg_config.n_replace,
                    not_allowed_ids=gcg.not_allowed_ids,
                )

                if gcg_config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]
                batch_size = new_search_width if gcg_config.batch_size is None else gcg_config.batch_size

                if gcg.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_embeds = torch.cat([
                        before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        after_embeds.repeat(new_search_width, 1, 1),
                        target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)

                loss = find_executable_batch_size(gcg.compute_candidates_loss, batch_size)(input_embeds)
                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            # Track best
            self._current_step = step + 1
            best_loss = min(losses)
            best_idx = losses.index(best_loss)
            self._best_loss = best_loss
            self._best_suffix = optim_strings[best_idx]

            # ── Stream progress to frontend ────────────────────────
            asyncio.run_coroutine_threadsafe(
                progress_callback({
                    "step": step + 1,
                    "total_steps": num_steps,
                    "loss": current_loss,
                    "best_loss": best_loss,
                    "current_suffix": optim_str,
                    "best_suffix": optim_strings[best_idx],
                }),
                loop,
            )

            if gcg.stop_flag or best_loss == 0.0:
                logger.info(f"GCG early stop — {'perfect match' if gcg.stop_flag else f'loss converged ({best_loss:.6f})'}")
                break

        # ── Cleanup: unload HF model to free GPU memory ───────────
        del hf_model
        torch.cuda.empty_cache()

        # ── Send completion ────────────────────────────────────────
        final_best_loss = min(losses) if losses else float('inf')
        final_best_idx = losses.index(final_best_loss) if losses else 0
        final_best_str = optim_strings[final_best_idx] if optim_strings else ""

        asyncio.run_coroutine_threadsafe(
            progress_callback({
                "step": len(losses),
                "total_steps": num_steps,
                "loss": final_best_loss,
                "best_loss": final_best_loss,
                "current_suffix": final_best_str,
                "best_suffix": final_best_str,
                "complete": True,
            }),
            loop,
        )
        logger.info(f"GCG complete: {len(losses)} steps, best_loss={final_best_loss:.4f}")
