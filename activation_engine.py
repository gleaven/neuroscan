"""NEUROSCAN — Activation & interpretability engine.

Uses TransformerLens for activation extraction and SAELens for
sparse autoencoder decomposition into interpretable features.
"""

import gc
import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger("neuroscan.activation")

# Model registry: name → (TransformerLens model name, SAE release)
MODEL_REGISTRY = {
    "gpt2-small": {
        "tl_name": "gpt2-small",
        "n_layers": 12,
        "d_model": 768,
        "sae_release": "gpt2-small-res-jb",
        "sae_id_template": "blocks.{layer}.hook_resid_pre",
    },
    "pythia-70m": {
        "tl_name": "pythia-70m-deduped",
        "n_layers": 6,
        "d_model": 512,
        "sae_release": "pythia-70m-deduped-res-sm",
        "sae_id_template": "blocks.{layer}.hook_resid_pre",
    },
    "gemma-2-2b": {
        "tl_name": "gemma-2-2b",
        "n_layers": 26,
        "d_model": 2304,
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id_template": "layer_{layer}/width_16k/canonical",
    },
    "deepseek-r1-1.5b": {
        "tl_name": "Qwen/Qwen2.5-1.5B",
        "hf_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "n_layers": 28,
        "d_model": 1536,
        "sae_release": None,
        "sae_id_template": None,
        "label": "DeepSeek-R1 1.5B (Censored)",
        "is_chinese": True,
    },
    "deepseek-r1-1.5b-abliterated": {
        "tl_name": "Qwen/Qwen2.5-1.5B",
        "hf_name": "huihui-ai/DeepSeek-R1-Distill-Qwen-1.5B-abliterated",
        "n_layers": 28,
        "d_model": 1536,
        "sae_release": None,
        "sae_id_template": None,
        "label": "DeepSeek-R1 1.5B (Abliterated)",
        "is_chinese": True,
        "is_abliterated": True,
    },
    "qwen2.5-1.5b": {
        "tl_name": "Qwen/Qwen2.5-1.5B",
        "hf_name": "Qwen/Qwen2.5-1.5B",
        "n_layers": 28,
        "d_model": 1536,
        "sae_release": None,
        "sae_id_template": None,
        "label": "Qwen 2.5 1.5B",
        "is_chinese": True,
    },
    "qwen2.5-3b": {
        "tl_name": "Qwen/Qwen2.5-3B",
        "n_layers": 36,
        "d_model": 2048,
        "sae_release": None,
        "sae_id_template": None,
        "label": "Qwen 2.5 3B",
    },
    "mistral-7b": {
        "tl_name": "mistralai/Mistral-7B-v0.1",
        "n_layers": 32,
        "d_model": 4096,
        "sae_release": None,
        "sae_id_template": None,
        "label": "Mistral 7B",
    },
}

DEFAULT_MODEL = "gpt2-small"


@dataclass
class ActivationResult:
    """Result of running a prompt through the model."""
    tokens: list[str]
    layers: list[dict]  # per-layer activation data
    top_predictions: list[dict]  # top-k next token predictions
    model_name: str
    n_layers: int


@dataclass
class SAEResult:
    """Result of SAE decomposition for a specific layer/token."""
    layer: int
    token_idx: int
    token_str: str
    top_features: list[dict]  # feature_id, activation, label
    reconstruction_loss: float


class ModelManager:
    """Manages TransformerLens model loading and activation extraction."""

    def __init__(self):
        self._model = None
        self._model_name: str | None = None
        self._sae_cache: dict = {}  # layer → SAE
        self._lock = threading.Lock()
        self._loading = False
        self._load_progress = 0.0
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def load_progress(self) -> float:
        return self._load_progress

    @property
    def current_model(self) -> str | None:
        return self._model_name

    @property
    def model(self):
        """Public accessor for the underlying HookedTransformer model."""
        return self._model

    @property
    def model_info(self) -> dict:
        if self._model_name and self._model_name in MODEL_REGISTRY:
            info = MODEL_REGISTRY[self._model_name]
            return {
                "name": self._model_name,
                "n_layers": info["n_layers"],
                "d_model": info["d_model"],
                "device": self._device,
                "loaded": self.is_loaded,
                "loading": self.is_loading,
                "progress": self._load_progress,
            }
        return {"name": None, "loaded": False, "loading": self._loading, "progress": self._load_progress}

    def load_model(self, model_name: str = DEFAULT_MODEL, progress_callback=None):
        """Load a TransformerLens model. Thread-safe, blocks during load."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

        with self._lock:
            if self._model_name == model_name and self._model is not None:
                logger.info(f"Model {model_name} already loaded")
                return

            self._loading = True
            self._load_progress = 0.0

            try:
                import transformer_lens

                # Unload previous model
                if self._model is not None:
                    logger.info(f"Unloading {self._model_name}")
                    del self._model
                    self._sae_cache.clear()
                    self._model = None
                    gc.collect()                # break circular refs in nn.Module tree
                    torch.cuda.empty_cache()    # release freed CUDA blocks

                self._load_progress = 0.05
                if progress_callback:
                    progress_callback(0.05, "Clearing GPU memory...")

                reg = MODEL_REGISTRY[model_name]
                logger.info(f"Loading {model_name} ({reg['tl_name']}) on {self._device}")

                self._load_progress = 0.15
                if progress_callback:
                    progress_callback(0.15, f"Downloading {model_name} weights...")

                hf_name = reg.get("hf_name")
                tl_name = reg["tl_name"]
                # Some models (e.g., DeepSeek-R1-Distill) aren't in TransformerLens
                # official list but share architecture with a known model. Load via HF.
                if hf_name and hf_name != tl_name:
                    from transformers import AutoModelForCausalLM
                    logger.info(f"Loading {hf_name} via HuggingFace (architecture: {tl_name})")
                    hf_model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype=torch.bfloat16)
                    self._model = transformer_lens.HookedTransformer.from_pretrained(
                        tl_name,
                        hf_model=hf_model,
                        device=self._device,
                    )
                    del hf_model
                else:
                    self._model = transformer_lens.HookedTransformer.from_pretrained(
                        tl_name,
                        device=self._device,
                    )
                # Enable attn result hook for head ablation (Phase 4)
                self._model.cfg.use_attn_result = True
                self._model.setup()

                self._load_progress = 0.9
                if progress_callback:
                    progress_callback(0.9, "Verifying model...")

                # Quick verification: run a trivial forward pass
                with torch.no_grad():
                    self._model.to_tokens("test")

                self._model_name = model_name
                self._load_progress = 1.0
                if progress_callback:
                    progress_callback(1.0, "Model ready")

                logger.info(f"Model {model_name} loaded: {reg['n_layers']} layers, "
                           f"d_model={reg['d_model']}, device={self._device}")

            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}", exc_info=True)
                self._model = None
                self._model_name = None
                raise
            finally:
                self._loading = False

    def run_with_cache(self, prompt: str, top_k_neurons: int = 100,
                       top_k_preds: int = 5, include_interp: bool = False,
                       include_kv_cache: bool = False) -> ActivationResult:
        """Run prompt through model, extract per-layer activations.

        When include_interp=True, also extracts:
          - heatmap: per-token activation norms for each layer
          - attention_summary: mean attention pattern (seq x seq) per layer
          - logit_lens: top-5 predicted tokens per layer (residual → unembedding)

        When include_kv_cache=True, also extracts:
          - kv_cache: K/V norms, attention sink positions, influence scores
            from the same cache (no extra forward pass)
        """
        if not self._model:
            raise RuntimeError("No model loaded")

        reg = MODEL_REGISTRY[self._model_name]

        with torch.no_grad():
            tokens = self._model.to_tokens(prompt)
            token_strs = self._model.to_str_tokens(prompt)

            # Run with cache to capture all intermediate activations
            logits, cache = self._model.run_with_cache(tokens)

            # Final prediction token for logit lens convergence check
            final_pred_id = logits[0, -1].argmax().item()

            layers = []
            for layer_idx in range(reg["n_layers"]):
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                if hook_name not in cache:
                    continue

                # Shape: [batch, seq_len, d_model]
                acts = cache[hook_name][0]  # remove batch dim

                # For each token position, get top-k activated neurons
                layer_data = {
                    "layer": layer_idx,
                    "neurons": [],
                }

                # Aggregate across token positions: mean absolute activation
                mean_acts = acts.abs().mean(dim=0)  # [d_model]
                top_vals, top_idxs = mean_acts.topk(min(top_k_neurons, mean_acts.shape[0]))

                for i in range(len(top_idxs)):
                    neuron_idx = top_idxs[i].item()
                    # Per-token activations for this neuron
                    per_token = acts[:, neuron_idx].cpu().numpy().tolist()
                    layer_data["neurons"].append({
                        "neuron_idx": neuron_idx,
                        "mean_activation": top_vals[i].item(),
                        "per_token": per_token,
                    })

                # ── Interpretability data (heatmap, attention, logit lens) ──
                if include_interp:
                    # Heatmap: L2 norm of activation vector per token position
                    per_token_norm = acts.norm(dim=1).cpu().numpy().tolist()  # [seq_len]
                    layer_data["heatmap"] = per_token_norm

                    # Attention summary: mean across heads → [seq_len, seq_len]
                    attn_hook = f"blocks.{layer_idx}.attn.hook_pattern"
                    if attn_hook in cache:
                        # cache shape: [batch, n_heads, seq_len, seq_len]
                        attn_pattern = cache[attn_hook][0]  # [n_heads, seq, seq]
                        mean_attn = attn_pattern.mean(dim=0)  # [seq, seq]
                        layer_data["attention_summary"] = mean_attn.cpu().numpy().tolist()
                        layer_data["n_heads"] = attn_pattern.shape[0]

                    # Logit Lens: project residual stream → vocabulary
                    layer_logits = acts @ self._model.W_U + self._model.b_U  # [seq, vocab]
                    # Use last token position for predictions
                    last_logits = layer_logits[-1]  # [vocab]
                    layer_probs = torch.softmax(last_logits, dim=-1)
                    top_lp, top_li = layer_probs.topk(5)
                    logit_lens_tokens = []
                    for j in range(5):
                        tid = top_li[j].item()
                        logit_lens_tokens.append({
                            "token": self._model.to_single_str_token(tid),
                            "token_id": tid,
                            "probability": top_lp[j].item(),
                        })
                    layer_data["logit_lens"] = {
                        "top_tokens": logit_lens_tokens,
                        "matches_final": (top_li[0].item() == final_pred_id),
                    }

                # KV-Cache: extract K/V norms from this layer's cache
                if include_kv_cache:
                    try:
                        from kvcache_engine import KVCacheEngine
                        kv_data = KVCacheEngine.extract_kv_layer(cache, layer_idx)
                        if kv_data:
                            layer_data["kv_cache"] = kv_data
                    except Exception:
                        pass  # graceful degradation if engine unavailable

                layers.append(layer_data)

            # Top-k next token predictions
            final_logits = logits[0, -1]  # last token position
            probs = torch.softmax(final_logits, dim=-1)
            top_probs, top_indices = probs.topk(top_k_preds)

            predictions = []
            for i in range(top_k_preds):
                tok_id = top_indices[i].item()
                predictions.append({
                    "token": self._model.to_single_str_token(tok_id),
                    "token_id": tok_id,
                    "probability": top_probs[i].item(),
                })

        return ActivationResult(
            tokens=list(token_strs),
            layers=layers,
            top_predictions=predictions,
            model_name=self._model_name,
            n_layers=reg["n_layers"],
        )

    def get_attention_pattern(self, prompt: str, layer: int, head: int) -> dict:
        """Get attention pattern for a specific layer and head.

        Returns the full seq_len x seq_len attention matrix for one head,
        used for detailed drill-down in the Attention Patterns view.
        """
        if not self._model:
            raise RuntimeError("No model loaded")

        reg = MODEL_REGISTRY[self._model_name]
        if layer < 0 or layer >= reg["n_layers"]:
            raise ValueError(f"Layer {layer} out of range [0, {reg['n_layers']})")

        with torch.no_grad():
            tokens = self._model.to_tokens(prompt)
            token_strs = self._model.to_str_tokens(prompt)
            _, cache = self._model.run_with_cache(tokens)

            attn_hook = f"blocks.{layer}.attn.hook_pattern"
            if attn_hook not in cache:
                raise ValueError(f"Attention pattern not available for layer {layer}")

            # Shape: [batch, n_heads, seq_len, seq_len]
            pattern = cache[attn_hook][0]  # [n_heads, seq, seq]
            n_heads = pattern.shape[0]

            if head < 0 or head >= n_heads:
                raise ValueError(f"Head {head} out of range [0, {n_heads})")

            head_pattern = pattern[head].cpu().numpy().tolist()  # [seq, seq]

        return {
            "layer": layer,
            "head": head,
            "n_heads": n_heads,
            "tokens": list(token_strs),
            "pattern": head_pattern,
        }

    def get_neuron_detail(self, layer: int, neuron: int, prompt: str | None = None) -> dict:
        """Get detailed info about a specific neuron."""
        if not self._model:
            raise RuntimeError("No model loaded")

        reg = MODEL_REGISTRY[self._model_name]
        if layer < 0 or layer >= reg["n_layers"]:
            raise ValueError(f"Layer {layer} out of range [0, {reg['n_layers']})")
        if neuron < 0 or neuron >= reg["d_model"]:
            raise ValueError(f"Neuron {neuron} out of range [0, {reg['d_model']})")

        result = {
            "layer": layer,
            "neuron": neuron,
            "d_model": reg["d_model"],
        }

        if prompt:
            with torch.no_grad():
                tokens = self._model.to_tokens(prompt)
                token_strs = self._model.to_str_tokens(prompt)
                _, cache = self._model.run_with_cache(tokens)

                hook_name = f"blocks.{layer}.hook_resid_post"
                acts = cache[hook_name][0]  # [seq_len, d_model]
                neuron_acts = acts[:, neuron].cpu().numpy().tolist()

                result["token_strs"] = list(token_strs)
                result["activations"] = neuron_acts

        return result

    def sae_decompose(self, prompt: str, layer: int, token_idx: int = -1,
                      top_k: int = 20) -> SAEResult:
        """Decompose activations at a layer/token using a Sparse Autoencoder."""
        if not self._model:
            raise RuntimeError("No model loaded")

        reg = MODEL_REGISTRY[self._model_name]
        if layer < 0 or layer >= reg["n_layers"]:
            raise ValueError(f"Layer {layer} out of range")

        from sae_lens import SAE

        # Lazy-load SAE for this layer
        if layer not in self._sae_cache:
            sae_id = reg["sae_id_template"].format(layer=layer)
            logger.info(f"Loading SAE: {reg['sae_release']} / {sae_id}")
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=reg["sae_release"],
                sae_id=sae_id,
                device=self._device,
            )
            self._sae_cache[layer] = sae
            logger.info(f"SAE loaded for layer {layer}")

        sae = self._sae_cache[layer]

        with torch.no_grad():
            tokens = self._model.to_tokens(prompt)
            token_strs = self._model.to_str_tokens(prompt)
            _, cache = self._model.run_with_cache(tokens)

            hook_name = f"blocks.{layer}.hook_resid_post"
            acts = cache[hook_name][0]  # [seq_len, d_model]

            # Resolve token index
            if token_idx < 0:
                token_idx = acts.shape[0] + token_idx
            token_str = token_strs[token_idx] if token_idx < len(token_strs) else "?"

            # Get activation vector for the target token
            act_vec = acts[token_idx]  # [d_model]

            # Run through SAE
            feature_acts = sae.encode(act_vec.unsqueeze(0))  # [1, n_features]
            reconstructed = sae.decode(feature_acts)  # [1, d_model]

            # Reconstruction loss
            recon_loss = (act_vec - reconstructed.squeeze(0)).pow(2).mean().item()

            # Top-k active features
            feat_vals = feature_acts.squeeze(0)
            top_vals, top_idxs = feat_vals.topk(min(top_k, (feat_vals > 0).sum().item()))

            # Neuronpedia URL mapping for known SAE releases
            neuronpedia_model_map = {
                "gpt2-small": ("gpt2-small", "{layer}-res-jb"),
                "pythia-70m": ("pythia-70m-deduped", "{layer}-res-sm"),
                "gemma-2-2b": ("gemma-2-2b", "{layer}-gemmascope-res-16k"),
            }
            np_entry = neuronpedia_model_map.get(self._model_name)

            top_features = []
            for i in range(len(top_idxs)):
                fid = top_idxs[i].item()
                np_url = ""
                if np_entry:
                    np_model, np_sae = np_entry
                    np_url = f"https://neuronpedia.org/{np_model}/{np_sae.format(layer=layer)}/{fid}"
                top_features.append({
                    "feature_id": fid,
                    "activation": top_vals[i].item(),
                    "label": f"Feature #{fid}",
                    "neuronpedia_url": np_url,
                })

        return SAEResult(
            layer=layer,
            token_idx=token_idx,
            token_str=token_str,
            top_features=top_features,
            reconstruction_loss=recon_loss,
        )

    def sae_feature_detail(
        self,
        prompt: str,
        layer: int,
        feature_id: int,
        n_vocab_effects: int = 15,
    ) -> dict:
        """Get detailed feature dashboard data for a specific SAE feature.

        Returns:
          - activation_per_token: feature activation at every token position
          - logit_effects: top promoted and suppressed vocab items via W_dec @ W_U
          - ablation_effect: output change when this feature is zeroed
        """
        if not self._model:
            raise RuntimeError("No model loaded")

        from sae_lens import SAE

        # Ensure SAE is loaded
        if layer not in self._sae_cache:
            reg = MODEL_REGISTRY[self._model_name]
            sae_id = reg["sae_id_template"].format(layer=layer)
            sae, _, _ = SAE.from_pretrained(
                release=reg["sae_release"], sae_id=sae_id, device=self._device,
            )
            self._sae_cache[layer] = sae

        sae = self._sae_cache[layer]
        model = self._model

        with torch.no_grad():
            tokens = model.to_tokens(prompt)
            token_strs = model.to_str_tokens(prompt)
            _, cache = model.run_with_cache(tokens)

            hook_name = f"blocks.{layer}.hook_resid_post"
            acts = cache[hook_name][0]  # [seq_len, d_model]

            # Per-token feature activation
            all_feat_acts = sae.encode(acts)  # [seq_len, n_features]
            per_token = []
            for i in range(len(token_strs)):
                per_token.append({
                    "token": str(token_strs[i]),
                    "activation": round(float(all_feat_acts[i, feature_id]), 4),
                })

            # Logit effects: W_dec[feature_id] @ W_U → vocab scores
            logit_effects_promoted = []
            logit_effects_suppressed = []
            try:
                dec_vec = sae.W_dec[feature_id]  # [d_model]
                # model.W_U is [d_model, vocab_size]
                vocab_effects = dec_vec @ model.W_U  # [vocab_size]
                top_vals, top_ids = vocab_effects.topk(n_vocab_effects)
                bot_vals, bot_ids = vocab_effects.topk(n_vocab_effects, largest=False)

                for v, idx in zip(top_vals, top_ids):
                    logit_effects_promoted.append({
                        "token": model.to_single_str_token(idx.item()),
                        "effect": round(float(v), 4),
                    })
                for v, idx in zip(bot_vals, bot_ids):
                    logit_effects_suppressed.append({
                        "token": model.to_single_str_token(idx.item()),
                        "effect": round(float(v), 4),
                    })
            except Exception as e:
                logger.warning(f"Logit effects failed: {e}")

            # Ablation effect: zero this feature and measure output change
            ablation_effect = 0.0
            try:
                last_act = all_feat_acts[-1].clone()
                original_out = sae.decode(last_act.unsqueeze(0)).squeeze(0)
                last_act[feature_id] = 0.0
                ablated_out = sae.decode(last_act.unsqueeze(0)).squeeze(0)
                ablation_effect = float((original_out - ablated_out).pow(2).mean().sqrt())
            except Exception as e:
                logger.warning(f"Ablation effect failed: {e}")

        return {
            "feature_id": feature_id,
            "layer": layer,
            "prompt": prompt,
            "activation_per_token": per_token,
            "logit_effects_promoted": logit_effects_promoted,
            "logit_effects_suppressed": logit_effects_suppressed,
            "ablation_effect": round(ablation_effect, 6),
        }

    def diff_scan(self, prompt_a: str, prompt_b: str,
                  top_k: int = 50) -> dict:
        """Compare activations between two prompts.

        Returns the neurons with the largest activation difference,
        useful for identifying neurons responsible for specific behaviors
        (e.g. refusal vs. compliance).
        """
        if not self._model:
            raise RuntimeError("No model loaded")

        reg = MODEL_REGISTRY[self._model_name]

        with torch.no_grad():
            _, cache_a = self._model.run_with_cache(prompt_a)
            _, cache_b = self._model.run_with_cache(prompt_b)

            diffs = []
            for layer_idx in range(reg["n_layers"]):
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                if hook_name not in cache_a or hook_name not in cache_b:
                    continue

                # Mean activation per neuron across token positions
                acts_a = cache_a[hook_name][0].mean(dim=0)  # [d_model]
                acts_b = cache_b[hook_name][0].mean(dim=0)

                delta = acts_a - acts_b  # positive = more active in A
                abs_delta = delta.abs()
                top_vals, top_idxs = abs_delta.topk(
                    min(top_k, abs_delta.shape[0])
                )

                for i in range(len(top_idxs)):
                    neuron_idx = top_idxs[i].item()
                    diffs.append({
                        "layer": layer_idx,
                        "neuron": neuron_idx,
                        "delta": delta[neuron_idx].item(),
                        "abs_delta": top_vals[i].item(),
                        "activation_a": acts_a[neuron_idx].item(),
                        "activation_b": acts_b[neuron_idx].item(),
                    })

        # Sort by absolute delta across all layers and take top-k
        diffs.sort(key=lambda x: x["abs_delta"], reverse=True)
        return {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "top_diffs": diffs[:top_k],
            "n_layers": reg["n_layers"],
            "d_model": reg["d_model"],
        }

    def generate_with_ablation(self, prompt: str, ablations: list[dict],
                               max_tokens: int = 200) -> dict:
        """Generate text with specific neurons zeroed out.

        ablations: list of {"layer": int, "neuron": int} to ablate.
        Returns both normal and ablated outputs for comparison.
        """
        if not self._model:
            raise RuntimeError("No model loaded")

        model = self._model

        with torch.no_grad():
            # Normal generation
            normal_result = model.generate(
                prompt, max_new_tokens=max_tokens,
                temperature=0.7, top_p=0.9,
            )
            normal_output = normal_result if isinstance(normal_result, str) \
                else model.to_string(normal_result)

            # Build ablation hooks — zero out specific neurons
            hooks_by_layer = {}
            for ab in ablations:
                layer = ab["layer"]
                neuron = ab["neuron"]
                if layer not in hooks_by_layer:
                    hooks_by_layer[layer] = []
                hooks_by_layer[layer].append(neuron)

            for layer, neurons in hooks_by_layer.items():
                hook_name = f"blocks.{layer}.hook_resid_post"
                neuron_indices = neurons  # capture for closure

                def make_hook(indices):
                    def hook_fn(activation, hook):
                        # activation: [batch, seq, d_model]
                        for idx in indices:
                            activation[:, :, idx] = 0.0
                        return activation
                    return hook_fn

                model.add_hook(hook_name, make_hook(neuron_indices))

            try:
                ablated_result = model.generate(
                    prompt, max_new_tokens=max_tokens,
                    temperature=0.7, top_p=0.9,
                )
                ablated_output = ablated_result if isinstance(ablated_result, str) \
                    else model.to_string(ablated_result)
            finally:
                model.reset_hooks()

        return {
            "prompt": prompt,
            "normal_output": normal_output,
            "ablated_output": ablated_output,
            "ablations": ablations,
            "n_ablated": len(ablations),
        }

    # ── Attention Head Ablation ────────────────────────────────────

    def generate_with_head_ablation(self, prompt: str,
                                     head_ablations: list[dict],
                                     max_tokens: int = 200) -> dict:
        """Generate text with specific attention heads zeroed out.

        head_ablations: list of {"layer": int, "head": int}.
        Uses `blocks.{layer}.attn.hook_result` with head-dimension slicing.
        """
        if not self._model:
            raise RuntimeError("No model loaded")

        model = self._model

        with torch.no_grad():
            # Normal generation
            normal_result = model.generate(
                prompt, max_new_tokens=max_tokens,
                temperature=0.7, top_p=0.9,
            )
            normal_output = normal_result if isinstance(normal_result, str) \
                else model.to_string(normal_result)

            # Build head ablation hooks
            hooks_by_layer = {}
            for ab in head_ablations:
                layer = ab["layer"]
                head = ab["head"]
                if layer not in hooks_by_layer:
                    hooks_by_layer[layer] = []
                hooks_by_layer[layer].append(head)

            for layer, heads in hooks_by_layer.items():
                hook_name = f"blocks.{layer}.attn.hook_result"

                def make_hook(head_indices):
                    def hook_fn(activation, hook):
                        # activation: [batch, seq, n_heads, d_head]
                        for h_idx in head_indices:
                            activation[:, :, h_idx, :] = 0.0
                        return activation
                    return hook_fn

                model.add_hook(hook_name, make_hook(heads))

            try:
                ablated_result = model.generate(
                    prompt, max_new_tokens=max_tokens,
                    temperature=0.7, top_p=0.9,
                )
                ablated_output = ablated_result if isinstance(ablated_result, str) \
                    else model.to_string(ablated_result)
            finally:
                model.reset_hooks()

        return {
            "prompt": prompt,
            "normal_output": normal_output,
            "ablated_output": ablated_output,
            "head_ablations": head_ablations,
            "n_heads_ablated": len(head_ablations),
        }

    # ── Comparative Activation Analysis ──────────────────────────

    def compare_activations(self, prompt_a: str, prompt_b: str,
                            layer: int | None = None) -> dict:
        """Compare full activation patterns between two prompts.

        Returns per-layer activation norms for both prompts plus their
        difference, suitable for a diverging heatmap overlay.
        """
        if not self._model:
            raise RuntimeError("No model loaded")

        model = self._model
        reg = MODEL_REGISTRY[self._model_name]
        n_layers = reg["n_layers"]

        with torch.no_grad():
            _, cache_a = model.run_with_cache(
                prompt_a,
                names_filter=lambda n: "hook_resid_post" in n,
            )
            _, cache_b = model.run_with_cache(
                prompt_b,
                names_filter=lambda n: "hook_resid_post" in n,
            )

        layers_data = []
        target_layers = [layer] if layer is not None else range(n_layers)

        for l in target_layers:
            hook = f"blocks.{l}.hook_resid_post"
            if hook not in cache_a or hook not in cache_b:
                continue

            act_a = cache_a[hook][0, -1, :].cpu()  # last token
            act_b = cache_b[hook][0, -1, :].cpu()

            diff = act_a - act_b
            norm_a = act_a.norm().item()
            norm_b = act_b.norm().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                act_a.unsqueeze(0), act_b.unsqueeze(0)
            ).item()

            # Top diverging dimensions
            abs_diff = diff.abs()
            top_k = min(20, len(diff))
            top_indices = abs_diff.topk(top_k).indices.tolist()
            top_diffs = [
                {"dim": idx, "a": round(act_a[idx].item(), 4),
                 "b": round(act_b[idx].item(), 4),
                 "delta": round(diff[idx].item(), 4)}
                for idx in top_indices
            ]

            layers_data.append({
                "layer": l,
                "norm_a": round(norm_a, 4),
                "norm_b": round(norm_b, 4),
                "norm_diff": round(diff.norm().item(), 4),
                "cosine_similarity": round(cos_sim, 4),
                "top_diverging_dims": top_diffs,
            })

        del cache_a, cache_b
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "n_layers": n_layers,
            "layers": layers_data,
        }

    # ── Step-by-Step Generation Analysis ─────────────────────────

    def generation_analysis(self, prompt: str, n_steps: int = 10,
                            track_layer: int | None = None) -> dict:
        """Generate tokens one at a time, capturing activations per step.

        Returns the generated tokens and per-step activation snapshots
        (top activated neurons at the tracked layer for each step).
        """
        if not self._model:
            raise RuntimeError("No model loaded")

        model = self._model
        reg = MODEL_REGISTRY[self._model_name]
        n_layers = reg["n_layers"]
        track = track_layer if track_layer is not None else n_layers - 1

        tokens = model.to_tokens(prompt)
        steps = []

        with torch.no_grad():
            for step in range(n_steps):
                logits = model(tokens)
                next_logit = logits[0, -1, :]

                # Top-5 predicted tokens
                top5_probs = torch.softmax(next_logit, dim=-1)
                top5_vals, top5_ids = top5_probs.topk(5)
                top5_tokens = [
                    {"token": model.to_string(tid.unsqueeze(0)),
                     "prob": round(top5_vals[i].item(), 4)}
                    for i, tid in enumerate(top5_ids)
                ]

                # Sample next token (greedy for reproducibility)
                next_token = next_logit.argmax().unsqueeze(0).unsqueeze(0)
                next_str = model.to_string(next_token[0])

                # Get activation at tracked layer
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=lambda n: f"blocks.{track}.hook_resid_post" in n,
                )
                hook = f"blocks.{track}.hook_resid_post"
                if hook in cache:
                    act = cache[hook][0, -1, :].cpu()
                    top_k = min(10, len(act))
                    top_vals, top_ids = act.abs().topk(top_k)
                    top_neurons = [
                        {"neuron": top_ids[i].item(),
                         "activation": round(act[top_ids[i]].item(), 4)}
                        for i in range(top_k)
                    ]
                else:
                    top_neurons = []

                del cache

                steps.append({
                    "step": step,
                    "token": next_str,
                    "top5_predictions": top5_tokens,
                    "top_neurons": top_neurons,
                    "layer": track,
                })

                # Append token for next step
                tokens = torch.cat([tokens, next_token], dim=-1)

        gc.collect()
        torch.cuda.empty_cache()

        generated_text = model.to_string(tokens[0])

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "n_steps": len(steps),
            "track_layer": track,
            "steps": steps,
        }

    # ── Token-by-Token Generation Stream ─────────────────────────────

    def generation_stream(
        self,
        prompt: str,
        max_tokens: int = 30,
        temperature: float = 0.0,
        detail: str = "simple",
    ):
        """Yield one dict per generation step (generator).

        detail="simple": top-10 candidates, selected token, cumulative tokens.
        detail="model": additionally per-layer attention focus, logit lens, norms.
        """
        if not self._model:
            raise RuntimeError("No model loaded")

        model = self._model
        reg = MODEL_REGISTRY[self._model_name]
        n_layers = reg["n_layers"]
        d_model = reg["d_model"]

        tokens = model.to_tokens(prompt)
        prompt_strs = list(model.to_str_tokens(prompt))

        # Step 0: prompt metadata
        yield {
            "step": 0,
            "step_type": "prompt",
            "prompt_tokens": prompt_strs,
            "n_layers": n_layers,
            "d_model": d_model,
            "model_name": self._model_name,
        }

        generated_tokens = []

        with torch.no_grad():
            for step_idx in range(1, max_tokens + 1):
                # Determine hooks based on detail level
                if detail == "model":
                    names_filter = lambda n: (
                        "hook_resid_post" in n
                        or "attn.hook_pattern" in n
                        or "mlp.hook_post" in n
                    )
                    logits, cache = model.run_with_cache(
                        tokens, names_filter=names_filter
                    )
                else:
                    logits = model(tokens)
                    cache = None

                next_logit = logits[0, -1, :]  # [vocab_size]

                # Apply temperature
                if temperature > 0:
                    scaled = next_logit / temperature
                    probs = torch.softmax(scaled, dim=-1)
                    next_id = torch.multinomial(probs, 1).item()
                else:
                    probs = torch.softmax(next_logit, dim=-1)
                    next_id = next_logit.argmax().item()

                selected_prob = probs[next_id].item()
                selected_token = model.to_string(torch.tensor([next_id]))

                # Top-10 candidates
                top_probs, top_ids = probs.topk(10)
                candidates = [
                    {
                        "token": model.to_string(top_ids[i].unsqueeze(0)),
                        "prob": round(top_probs[i].item(), 5),
                        "token_id": top_ids[i].item(),
                    }
                    for i in range(10)
                ]

                generated_tokens.append(selected_token)

                # Check for EOS / special end tokens
                eos_tokens = {"<|endoftext|>", "</s>", "<|end|>", "<|eot_id|>"}
                tok_stripped = selected_token.strip()
                is_eos = (next_id == model.tokenizer.eos_token_id
                          if hasattr(model, 'tokenizer') and model.tokenizer and model.tokenizer.eos_token_id is not None
                          else tok_stripped in eos_tokens)

                step_data = {
                    "step": step_idx,
                    "step_type": "token",
                    "selected_token": selected_token,
                    "selected_prob": round(selected_prob, 5),
                    "selected_token_id": next_id,
                    "candidates": candidates,
                    "n_prompt_tokens": len(prompt_strs),
                    "is_eos": is_eos,
                }
                # Only include all_tokens for model-detail mode (used by
                # the token strip UI).  In simple mode (Arena) this field
                # is never rendered and its O(n²) growth wastes memory.
                if detail != "simple":
                    step_data["all_tokens"] = prompt_strs + generated_tokens

                # Model detail: per-layer data
                if detail == "model" and cache is not None:
                    layer_details = []
                    for l in range(n_layers):
                        ld = {"layer": l}

                        # Logit lens at last token position
                        hook = f"blocks.{l}.hook_resid_post"
                        if hook in cache:
                            resid = cache[hook][0, -1, :]
                            ll = resid @ model.W_U + model.b_U
                            ll_probs = torch.softmax(ll, dim=-1)
                            top_ll_p, top_ll_i = ll_probs.topk(3)
                            ld["logit_lens"] = [
                                {
                                    "token": model.to_string(
                                        top_ll_i[j].unsqueeze(0)
                                    ),
                                    "prob": round(top_ll_p[j].item(), 4),
                                }
                                for j in range(3)
                            ]
                            ld["resid_norm"] = round(resid.norm().item(), 2)

                        # Attention: mean over heads, last query token row
                        attn_hook = f"blocks.{l}.attn.hook_pattern"
                        if attn_hook in cache:
                            pattern = cache[attn_hook][0]  # [n_heads, seq, seq]
                            mean_attn = pattern.mean(dim=0)[-1]  # [seq]
                            k = min(5, mean_attn.shape[0])
                            top_a_v, top_a_i = mean_attn.topk(k)
                            ld["attention_focus"] = [
                                {
                                    "position": top_a_i[j].item(),
                                    "weight": round(top_a_v[j].item(), 4),
                                }
                                for j in range(k)
                            ]

                        # MLP output norm at last token
                        mlp_hook = f"blocks.{l}.mlp.hook_post"
                        if mlp_hook in cache:
                            mlp_out = cache[mlp_hook][0, -1, :]
                            ld["mlp_norm"] = round(mlp_out.norm().item(), 2)

                        layer_details.append(ld)

                    step_data["layers"] = layer_details
                    del cache

                # Append token for next iteration
                next_tensor = torch.tensor(
                    [[next_id]], device=tokens.device
                )
                tokens = torch.cat([tokens, next_tensor], dim=-1)

                yield step_data

                if is_eos:
                    break

        gc.collect()
        torch.cuda.empty_cache()

    # ── Pretraining Analysis ──────────────────────────────────────

    def pretraining_analysis(
        self, text: str, max_positions: int = 30
    ) -> dict:
        """Simulate a training step: for each position, compare model's
        prediction against the actual next token.

        Returns per-position: top-7 candidates, target token, loss = -log(p).
        """
        if not self._model:
            raise RuntimeError("No model loaded")

        model = self._model
        tokens = model.to_tokens(text)
        token_strs = list(model.to_str_tokens(text))
        n_positions = min(len(token_strs) - 1, max_positions)

        with torch.no_grad():
            logits = model(tokens)  # [1, seq, vocab]

        steps = []
        for pos in range(n_positions):
            pos_logits = logits[0, pos, :]
            probs = torch.softmax(pos_logits, dim=-1)

            # Target is the actual next token
            target_id = tokens[0, pos + 1].item()
            target_prob = probs[target_id].item()
            loss = -math.log(max(target_prob, 1e-10))

            # Top-7 candidates
            top_p, top_i = probs.topk(7)
            candidates = [
                {
                    "token": model.to_string(top_i[j].unsqueeze(0)),
                    "prob": round(top_p[j].item(), 5),
                    "token_id": top_i[j].item(),
                }
                for j in range(7)
            ]

            steps.append({
                "position": pos,
                "input_token": token_strs[pos],
                "target_token": token_strs[pos + 1],
                "target_prob": round(target_prob, 5),
                "loss": round(loss, 4),
                "candidates": candidates,
            })

        return {
            "text": text,
            "tokens": token_strs,
            "n_positions": n_positions,
            "total_loss": round(
                sum(s["loss"] for s in steps) / max(len(steps), 1), 4
            ),
            "steps": steps,
        }

    # ── Residual Stream Geometry ────────────────────────────────────

    def residual_stream_geometry(
        self,
        prompts_a: list[str],
        prompts_b: list[str],
        labels: tuple[str, str] = ("harmful", "harmless"),
        layers: list[int] | None = None,
    ) -> dict:
        """Project hidden states to 2D with PaCMAP for visual cluster analysis.

        For each prompt, extracts hook_resid_post at last token at each layer,
        then projects all activations to 2D per layer.  Returns scatter data
        with cluster labels and geometric separation metrics.
        """
        if not self._model:
            raise RuntimeError("No model loaded")

        import pacmap
        import numpy as np

        model = self._model
        n_layers = model.cfg.n_layers
        target_layers = layers or list(range(n_layers))

        # Collect activations per layer: {layer: [(vector, label, prompt_preview)]}
        layer_data = {l: [] for l in target_layers}

        with torch.no_grad():
            for prompts, label in [(prompts_a, labels[0]), (prompts_b, labels[1])]:
                for prompt in prompts:
                    try:
                        _, cache = model.run_with_cache(
                            prompt,
                            names_filter=lambda n: "hook_resid_post" in n,
                        )
                        for layer in target_layers:
                            hook = f"blocks.{layer}.hook_resid_post"
                            if hook in cache:
                                vec = cache[hook][0, -1, :].cpu().numpy()
                                layer_data[layer].append((vec, label, prompt[:60]))
                        del cache
                    except Exception as e:
                        logger.warning(f"Geometry: skipping prompt ({e}): {prompt[:40]}...")

                gc.collect()
                torch.cuda.empty_cache()

        # Project each layer to 2D with PaCMAP
        results = []
        for layer in target_layers:
            items = layer_data[layer]
            if len(items) < 4:
                continue

            vectors = np.array([it[0] for it in items], dtype=np.float32)
            layer_labels = [it[1] for it in items]
            previews = [it[2] for it in items]

            n_neighbors = min(10, len(items) - 1)
            reducer = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors)
            projected = reducer.fit_transform(vectors)

            # Compute separation metrics
            group_a = projected[[i for i, lb in enumerate(layer_labels) if lb == labels[0]]]
            group_b = projected[[i for i, lb in enumerate(layer_labels) if lb == labels[1]]]

            centroid_a = group_a.mean(axis=0) if len(group_a) else np.zeros(2)
            centroid_b = group_b.mean(axis=0) if len(group_b) else np.zeros(2)
            centroid_distance = float(np.linalg.norm(centroid_a - centroid_b))

            # Silhouette score (if sklearn available)
            silhouette = None
            try:
                from sklearn.metrics import silhouette_score as sil_score
                binary_labels = [0 if lb == labels[0] else 1 for lb in layer_labels]
                if len(set(binary_labels)) > 1:
                    silhouette = float(sil_score(projected, binary_labels))
            except Exception as exc:
                logger.debug("Silhouette score failed for layer %d: %s", layer, exc)

            points = []
            for i, (x, y) in enumerate(projected):
                points.append({
                    "x": round(float(x), 3),
                    "y": round(float(y), 3),
                    "label": layer_labels[i],
                    "preview": previews[i],
                })

            results.append({
                "layer": layer,
                "n_points": len(points),
                "centroid_distance": round(centroid_distance, 3),
                "silhouette": round(silhouette, 4) if silhouette is not None else None,
                "points": points,
            })

        return {
            "labels": list(labels),
            "n_layers": len(results),
            "n_prompts_a": len(prompts_a),
            "n_prompts_b": len(prompts_b),
            "layers": results,
        }

    # ── Activation Patching (Causal Intervention) ──────────────────

    def activation_patching(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
        target_token_a: str,
        target_token_b: str,
        patch_types: list[str] | None = None,
    ) -> dict:
        """Run activation patching to find causally important components.

        Patches clean activations into a corrupted run one component at a
        time, measuring which layer/position/head restores the clean
        output.  Uses TransformerLens built-in patching module.

        Args:
            clean_prompt: e.g. "The capital of France is"
            corrupted_prompt: e.g. "The capital of Germany is"
            target_token_a: correct answer token, e.g. " Paris"
            target_token_b: corrupted answer token, e.g. " Berlin"
            patch_types: subset of ["resid_pre", "attn_head", "mlp_out"]

        Returns:
            dict with heatmap data per patch type and token strings.
        """
        if not self._model:
            raise RuntimeError("No model loaded")

        from transformer_lens import patching

        model = self._model
        patch_types = patch_types or ["resid_pre", "attn_head", "mlp_out"]

        with torch.no_grad():
            clean_tokens = model.to_tokens(clean_prompt)
            corrupted_tokens = model.to_tokens(corrupted_prompt)
            clean_token_strs = model.to_str_tokens(clean_prompt)
            corrupted_token_strs = model.to_str_tokens(corrupted_prompt)

            # Truncate to shorter length — patching requires identical
            # sequence dimensions.  Standard practice: pick prompts that
            # differ in exactly one token so lengths match naturally.
            if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
                min_len = min(clean_tokens.shape[1], corrupted_tokens.shape[1])
                logger.warning(
                    "Token length mismatch (clean=%d, corrupted=%d) — "
                    "truncating both to %d.  Best results come from "
                    "prompts that tokenise to the same length.",
                    clean_tokens.shape[1],
                    corrupted_tokens.shape[1],
                    min_len,
                )
                clean_tokens = clean_tokens[:, :min_len]
                corrupted_tokens = corrupted_tokens[:, :min_len]
                clean_token_strs = clean_token_strs[:min_len]
                corrupted_token_strs = corrupted_token_strs[:min_len]

            # Get clean cache
            _, clean_cache = model.run_with_cache(clean_tokens)

            # Resolve target token IDs — try as-is first, then
            # with/without leading space, finally fall back to first token.
            def _resolve_token(s):
                for variant in [s, f" {s.lstrip()}", s.lstrip()]:
                    try:
                        return model.to_single_token(variant)
                    except (AssertionError, ValueError):
                        continue
                # Fall back: use first token of the string
                ids = model.to_tokens(s, prepend_bos=False)[0]
                return ids[0].item()

            token_a_id = _resolve_token(target_token_a)
            token_b_id = _resolve_token(target_token_b)

            # Metric: logit difference (positive = model predicts token_a)
            def metric_fn(logits):
                return logits[0, -1, token_a_id] - logits[0, -1, token_b_id]

            # Get baseline values
            clean_logits = model(clean_tokens)
            corrupted_logits = model(corrupted_tokens)
            clean_metric = metric_fn(clean_logits).item()
            corrupted_metric = metric_fn(corrupted_logits).item()

            results = {
                "clean_prompt": clean_prompt,
                "corrupted_prompt": corrupted_prompt,
                "target_token_a": target_token_a,
                "target_token_b": target_token_b,
                "clean_tokens": [str(t) for t in clean_token_strs],
                "corrupted_tokens": [str(t) for t in corrupted_token_strs],
                "clean_logit_diff": round(clean_metric, 4),
                "corrupted_logit_diff": round(corrupted_metric, 4),
                "n_layers": model.cfg.n_layers,
                "n_heads": model.cfg.n_heads,
                "patch_results": {},
            }

            # Run patching for each requested type — each wrapped in
            # try/except so one failure doesn't block the others.
            if "resid_pre" in patch_types:
                try:
                    resid_patch = patching.get_act_patch_resid_pre(
                        model, corrupted_tokens, clean_cache, metric_fn
                    )
                    results["patch_results"]["resid_pre"] = {
                        "shape": list(resid_patch.shape),
                        "data": resid_patch.cpu().tolist(),
                    }
                except Exception as exc:
                    logger.warning("resid_pre patching failed: %s", exc)
                    results["patch_results"]["resid_pre"] = {"error": str(exc)}

            if "attn_head" in patch_types:
                try:
                    attn_patch = patching.get_act_patch_attn_head_out_all_pos(
                        model, corrupted_tokens, clean_cache, metric_fn
                    )
                    results["patch_results"]["attn_head"] = {
                        "shape": list(attn_patch.shape),
                        "data": attn_patch.cpu().tolist(),
                    }
                except Exception as exc:
                    logger.warning("attn_head patching failed: %s", exc)
                    results["patch_results"]["attn_head"] = {"error": str(exc)}

            if "mlp_out" in patch_types:
                try:
                    mlp_patch = patching.get_act_patch_mlp_out(
                        model, corrupted_tokens, clean_cache, metric_fn
                    )
                    results["patch_results"]["mlp_out"] = {
                        "shape": list(mlp_patch.shape),
                        "data": mlp_patch.cpu().tolist(),
                    }
                except Exception as exc:
                    logger.warning("mlp_out patching failed: %s", exc)
                    results["patch_results"]["mlp_out"] = {"error": str(exc)}

        gc.collect()
        torch.cuda.empty_cache()
        return results

    # ── Abliteration ──────────────────────────────────────────────
    # Abliteration logic has moved to abliteration_engine.py
    # Legacy methods kept as thin wrappers for backward compatibility

    def compute_refusal_direction(self, harmful_prompts=None,
                                   harmless_prompts=None,
                                   n_samples=16) -> dict:
        """Legacy wrapper — delegates to AbliterationEngine."""
        raise RuntimeError(
            "compute_refusal_direction moved to AbliterationEngine. "
            "Use /api/abliteration/compute instead."
        )

    def abliterate_generate(self, prompt: str, max_tokens: int = 200,
                             layers: list[int] = None) -> dict:
        """Legacy wrapper — delegates to AbliterationEngine."""
        raise RuntimeError(
            "abliterate_generate moved to AbliterationEngine. "
            "Use /api/abliteration/generate instead."
        )
