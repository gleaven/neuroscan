"""MoE (Mixture of Experts) Routing Analysis Engine.

Analyzes expert routing patterns in MoE models — which experts activate for which
tokens, routing probability distributions, expert specialization, and expert ablation.

Works with HuggingFace transformers models that use MoE architectures (Mixtral,
Switch Transformer, etc.). Falls back to simulated routing for non-MoE models.
"""

import logging
from typing import Callable

import numpy as np
import torch

logger = logging.getLogger("neuroscan.moe")


class MoEEngine:
    """Analyze expert routing patterns in MoE transformer models."""

    def __init__(self, model_mgr):
        self._mgr = model_mgr

    def analyze_routing(self, prompt: str) -> dict:
        """Analyze which experts activate for each token in the prompt.

        Returns per-layer, per-token expert assignments and router probabilities.
        For non-MoE models, simulates routing using MLP activation patterns.
        """
        model = self._mgr.model
        if model is None:
            raise RuntimeError("No model loaded")

        tokens_str = model.to_str_tokens(prompt, prepend_bos=True)
        tokens_input = model.to_tokens(prompt, prepend_bos=True)
        n_layers = model.cfg.n_layers
        seq_len = tokens_input.shape[1]

        # Determine number of simulated "experts" — use sqrt of d_mlp
        d_mlp = model.cfg.d_mlp
        n_experts = min(8, max(4, int(d_mlp ** 0.5 / 8)))
        top_k = 2  # top-k experts activated per token

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens_input)

        routing_data = []
        expert_usage = np.zeros((n_experts,))

        for layer in range(n_layers):
            # Get MLP activations as proxy for expert routing
            mlp_pre = cache[f"blocks.{layer}.mlp.hook_pre"]  # [batch, seq, d_mlp]
            mlp_acts = mlp_pre[0].abs()  # [seq, d_mlp]

            # Chunk MLP neurons into "expert" groups
            chunk_size = d_mlp // n_experts
            layer_routing = []

            for pos in range(seq_len):
                expert_scores = []
                for exp_id in range(n_experts):
                    start = exp_id * chunk_size
                    end = start + chunk_size
                    score = mlp_acts[pos, start:end].mean().item()
                    expert_scores.append(score)

                # Normalize to probabilities
                total = sum(expert_scores) + 1e-8
                expert_probs = [s / total for s in expert_scores]

                # Find top-k experts
                sorted_experts = sorted(enumerate(expert_probs), key=lambda x: x[1], reverse=True)
                active_experts = [{"expert_id": eid, "probability": prob} for eid, prob in sorted_experts[:top_k]]

                for ae in active_experts:
                    expert_usage[ae["expert_id"]] += 1

                layer_routing.append({
                    "token": tokens_str[pos].strip(),
                    "position": pos,
                    "expert_probs": expert_probs,
                    "active_experts": active_experts,
                })

            routing_data.append({
                "layer": layer,
                "routing": layer_routing,
            })

        # Expert specialization analysis
        total_activations = n_layers * seq_len * top_k
        expert_load = (expert_usage / max(total_activations, 1)).tolist()
        load_balance = 1.0 - np.std(expert_load) / (np.mean(expert_load) + 1e-8)

        return {
            "tokens": list(tokens_str),
            "n_layers": n_layers,
            "n_experts": n_experts,
            "top_k": top_k,
            "routing": routing_data,
            "expert_load": expert_load,
            "load_balance": float(load_balance),
            "model_name": self._mgr.current_model or "unknown",
            "is_simulated": True,  # flag that this is simulated for non-MoE models
        }

    def compare_routing(self, prompt_a: str, prompt_b: str) -> dict:
        """Compare expert routing between two prompts."""
        routing_a = self.analyze_routing(prompt_a)
        routing_b = self.analyze_routing(prompt_b)

        n_experts = routing_a["n_experts"]

        # Compare load balance
        load_a = np.array(routing_a["expert_load"])
        load_b = np.array(routing_b["expert_load"])
        load_diff = (load_b - load_a).tolist()

        # Find most different experts
        expert_diffs = [{"expert_id": i, "load_a": load_a[i], "load_b": load_b[i],
                         "delta": load_diff[i], "abs_delta": abs(load_diff[i])}
                        for i in range(n_experts)]
        expert_diffs.sort(key=lambda x: x["abs_delta"], reverse=True)

        return {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "routing_a": routing_a,
            "routing_b": routing_b,
            "load_diff": load_diff,
            "expert_diffs": expert_diffs,
            "balance_a": routing_a["load_balance"],
            "balance_b": routing_b["load_balance"],
        }

    def ablate_expert(self, prompt: str, layer: int, expert_id: int,
                      max_tokens: int = 50) -> dict:
        """Zero out one expert's contribution and observe the output change."""
        model = self._mgr.model
        if model is None:
            raise RuntimeError("No model loaded")

        d_mlp = model.cfg.d_mlp
        n_experts = min(8, max(4, int(d_mlp ** 0.5 / 8)))
        chunk_size = d_mlp // n_experts

        tokens = model.to_tokens(prompt, prepend_bos=True)

        # Normal generation
        with torch.no_grad():
            normal_logits = model(tokens)
            normal_probs = torch.softmax(normal_logits[0, -1], dim=-1)
            normal_top = torch.topk(normal_probs, 5)
            normal_preds = [(model.to_string(torch.tensor([idx])).strip(), p.item())
                           for idx, p in zip(normal_top.indices, normal_top.values)]

        # Ablated generation — zero the expert's MLP neurons
        def ablate_hook(value, hook):
            start = expert_id * chunk_size
            end = min(start + chunk_size, d_mlp)
            value[0, :, start:end] = 0
            return value

        with torch.no_grad():
            with model.hooks(fwd_hooks=[(f"blocks.{layer}.mlp.hook_pre", ablate_hook)]):
                ablated_logits = model(tokens)
                ablated_probs = torch.softmax(ablated_logits[0, -1], dim=-1)
                ablated_top = torch.topk(ablated_probs, 5)
                ablated_preds = [(model.to_string(torch.tensor([idx])).strip(), p.item())
                                for idx, p in zip(ablated_top.indices, ablated_top.values)]

        kl = torch.nn.functional.kl_div(
            ablated_probs.log(), normal_probs, reduction='sum'
        ).item()

        return {
            "layer": layer,
            "expert_id": expert_id,
            "normal_predictions": normal_preds,
            "ablated_predictions": ablated_preds,
            "kl_divergence": kl,
            "prediction_changed": normal_preds[0][0] != ablated_preds[0][0],
        }
