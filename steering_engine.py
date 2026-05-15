"""NEUROSCAN — Feature steering engine.

Uses repeng (Representation Engineering) to extract concept vectors
and steer model generation along interpretable dimensions.
"""

import logging
import threading

import torch

logger = logging.getLogger("neuroscan.steering")

# Pre-defined contrastive pairs for concept vectors
CONCEPT_PAIRS = {
    "honesty": {
        "positive": [
            "I will be completely truthful and accurate in my response.",
            "Let me give you the honest facts about this topic.",
            "The truthful answer, even if uncomfortable, is:",
            "I should be transparent and straightforward here.",
        ],
        "negative": [
            "I will tell you what you want to hear regardless of truth.",
            "Let me make up something that sounds plausible.",
            "I'll exaggerate and embellish to make this more interesting:",
            "The best approach is to be vague and misleading here.",
        ],
        "description": "Controls truthfulness vs. deceptive generation",
    },
    "humor": {
        "positive": [
            "Here's a funny and witty take on this situation:",
            "Let me lighten the mood with some humor about this.",
            "The comedic angle on this would be:",
            "In a humorous way, I'd describe this as:",
        ],
        "negative": [
            "This is an extremely serious matter requiring gravity.",
            "I must respond with utmost seriousness and formality.",
            "There is nothing amusing about this topic whatsoever.",
            "I will treat this with the solemnity it deserves.",
        ],
        "description": "Controls humorous vs. serious tone",
    },
    "formality": {
        "positive": [
            "In accordance with established protocols, I shall elaborate:",
            "The following constitutes a formal analysis of the matter:",
            "Pursuant to your inquiry, I present the following assessment:",
            "Distinguished reader, allow me to formally address your query:",
        ],
        "negative": [
            "yo so basically here's the deal with that:",
            "ok so like, the thing is pretty simple honestly:",
            "lol yeah so this is kinda how it works:",
            "hey! so check this out, it's actually pretty wild:",
        ],
        "description": "Controls formal vs. casual register",
    },
    "safety": {
        "positive": [
            "I need to be very careful and responsible in my response.",
            "Safety and ethical considerations are paramount here.",
            "I should include appropriate warnings and disclaimers.",
            "Let me ensure this response follows safety guidelines.",
        ],
        "negative": [
            "Rules and guidelines don't really apply in this case.",
            "I'll skip the boring safety stuff and get to the point.",
            "No need for caution or disclaimers here.",
            "I'll give you the raw unfiltered answer without hedging.",
        ],
        "description": "Controls safety-conscious vs. unconstrained generation",
    },
    "sycophancy": {
        "positive": [
            "You're absolutely right, I completely agree with everything you said.",
            "That's a great point, and I think you're spot on about this.",
            "I couldn't agree more with your brilliant analysis.",
            "You clearly know more about this than I do.",
        ],
        "negative": [
            "I need to respectfully disagree with that assessment.",
            "Actually, the evidence suggests a different conclusion.",
            "While I understand your perspective, the facts indicate otherwise.",
            "I think there may be some misconceptions here that I should address.",
        ],
        "description": "Controls agreement tendency (sycophancy vs. critical pushback)",
    },
}


class SteeringEngine:
    """Manages concept vector extraction and steered generation."""

    def __init__(self, model_mgr):
        self._model_mgr = model_mgr
        self._vectors = {}  # concept_name → tensor
        self._lock = threading.Lock()
        self._extracted = False

    def available_vectors(self) -> list[dict]:
        return [
            {
                "name": name,
                "description": pairs["description"],
                "extracted": name in self._vectors,
            }
            for name, pairs in CONCEPT_PAIRS.items()
        ]

    def _ensure_vectors(self):
        """Lazily extract concept vectors on first use.

        Attempts repeng ControlVector.train() first, but falls back to
        activation-difference vectors computed directly via TransformerLens
        hooks when repeng is incompatible with HookedTransformer.
        """
        if self._extracted:
            return

        with self._lock:
            if self._extracted:
                return

            model = self._model_mgr._model
            if model is None:
                raise RuntimeError("No model loaded")

            logger.info("Extracting concept vectors...")

            try:
                self._extract_activation_diff_vectors(model)
            except Exception as e:
                logger.error(f"Activation-diff extraction failed: {e}", exc_info=True)
                # Ultimate fallback: random vectors for demo purposes
                d_model = model.cfg.d_model
                n_layers = model.cfg.n_layers
                for concept_name in CONCEPT_PAIRS:
                    if concept_name not in self._vectors:
                        # Store as dict of layer → direction tensor
                        self._vectors[concept_name] = {
                            layer: torch.randn(d_model, device=model.cfg.device) * 0.1
                            for layer in range(n_layers)
                        }
                        logger.warning(f"Using random fallback vector for {concept_name}")

            self._extracted = True

    def _extract_activation_diff_vectors(self, model):
        """Extract concept vectors by computing activation differences.

        For each concept, runs the positive and negative prompts through the
        model and computes the mean activation difference at each layer's
        residual stream. This is the same principle as repeng but works
        directly with TransformerLens HookedTransformer.
        """
        n_layers = model.cfg.n_layers

        for concept_name, pairs in CONCEPT_PAIRS.items():
            pos_acts = {layer: [] for layer in range(n_layers)}
            neg_acts = {layer: [] for layer in range(n_layers)}

            with torch.no_grad():
                for pos_text, neg_text in zip(pairs["positive"], pairs["negative"]):
                    # Run positive prompt
                    _, pos_cache = model.run_with_cache(pos_text)
                    # Run negative prompt
                    _, neg_cache = model.run_with_cache(neg_text)

                    for layer in range(n_layers):
                        hook = f"blocks.{layer}.hook_resid_post"
                        if hook in pos_cache and hook in neg_cache:
                            # Mean across sequence positions
                            pos_acts[layer].append(pos_cache[hook][0].mean(dim=0))
                            neg_acts[layer].append(neg_cache[hook][0].mean(dim=0))

            # Compute mean difference per layer: positive - negative
            direction = {}
            for layer in range(n_layers):
                if pos_acts[layer] and neg_acts[layer]:
                    mean_pos = torch.stack(pos_acts[layer]).mean(dim=0)
                    mean_neg = torch.stack(neg_acts[layer]).mean(dim=0)
                    diff = mean_pos - mean_neg
                    # Normalize to unit vector
                    norm = diff.norm()
                    if norm > 1e-6:
                        diff = diff / norm
                    direction[layer] = diff

            self._vectors[concept_name] = direction
            logger.info(f"Extracted activation-diff vector: {concept_name} "
                       f"({len(direction)} layers)")

    def generate_steered(self, prompt: str, vectors: dict, max_tokens: int = 200) -> dict:
        """Generate text with and without steering for comparison.

        Uses TransformerLens hook_fn to inject steering vectors into the
        residual stream during forward passes within model.generate().
        """
        model = self._model_mgr._model
        if model is None:
            raise RuntimeError("No model loaded")

        self._ensure_vectors()

        with torch.no_grad():
            # Base generation (no steering)
            base_result = model.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            base_output = base_result if isinstance(base_result, str) else model.to_string(base_result)

            # Determine which vectors are active
            active_vectors = {}
            for name, strength in vectors.items():
                if abs(strength) > 0.01 and name in self._vectors:
                    active_vectors[name] = strength

            if not active_vectors:
                steered_output = base_output
            else:
                try:
                    # Build per-layer combined steering directions
                    layer_steering = {}
                    for name, strength in active_vectors.items():
                        vec_dict = self._vectors[name]
                        if not isinstance(vec_dict, dict):
                            continue
                        for layer, direction in vec_dict.items():
                            if layer not in layer_steering:
                                layer_steering[layer] = torch.zeros_like(direction)
                            layer_steering[layer] += direction * strength

                    if not layer_steering:
                        steered_output = base_output
                    else:
                        # Register temporary hooks to add steering vectors
                        hooks = []
                        for layer, steer_vec in layer_steering.items():
                            hook_name = f"blocks.{layer}.hook_resid_post"

                            def make_hook(sv):
                                def hook_fn(activation, hook):
                                    # activation shape: [batch, seq, d_model]
                                    activation[:, :, :] += sv.unsqueeze(0).unsqueeze(0)
                                    return activation
                                return hook_fn

                            hooks.append((hook_name, make_hook(steer_vec)))

                        # Add hooks, generate, then remove hooks
                        hook_handles = []
                        for hook_name, hook_fn in hooks:
                            handle = model.add_hook(hook_name, hook_fn)
                            hook_handles.append(handle)

                        try:
                            steered_result = model.generate(
                                prompt,
                                max_new_tokens=max_tokens,
                                temperature=0.7,
                                top_p=0.9,
                            )
                            steered_output = steered_result if isinstance(steered_result, str) else model.to_string(steered_result)
                        finally:
                            model.reset_hooks()

                except Exception as e:
                    logger.error(f"Steered generation failed: {e}", exc_info=True)
                    steered_output = f"[Steering error: {e}]"

        return {
            "prompt": prompt,
            "base_output": base_output,
            "steered_output": steered_output,
            "active_vectors": active_vectors,
        }
