"""NEUROSCAN — KV-Cache analysis engine.

Extracts Key/Value cache tensors from TransformerLens hooks and performs
security-relevant analyses: attention sink detection, cache influence scoring,
prefix poisoning simulation, quantization drift measurement, and cache
exhaustion projection.

The KV-cache is the dynamic working memory of transformer inference.  Unlike
model weights (static and signed), cached K/V tensors are computed from user
input with no integrity verification — making them a compelling adversarial
attack surface.
"""

import logging
import threading
from collections import Counter

import numpy as np
import torch

logger = logging.getLogger("neuroscan.kvcache")


class KVCacheEngine:
    """Analyzes KV-cache behavior for security and interpretability."""

    def __init__(self, model_mgr):
        self._mgr = model_mgr
        self._lock = threading.Lock()
        self._cancel = threading.Event()

    # ── Core extraction (piggybacked on existing run_with_cache) ────

    @staticmethod
    def extract_kv_layer(cache, layer_idx):
        """Extract KV norms and sink data for one layer from a cache dict.

        Called from activation_engine.run_with_cache() so no extra forward
        pass is needed.  TransformerLens hooks:
          cache["blocks.{L}.attn.hook_k"]  →  [batch, seq, n_heads, d_head]
          cache["blocks.{L}.attn.hook_v"]  →  [batch, seq, n_heads, d_head]
        """
        k_hook = f"blocks.{layer_idx}.attn.hook_k"
        v_hook = f"blocks.{layer_idx}.attn.hook_v"

        if k_hook not in cache or v_hook not in cache:
            return None

        k = cache[k_hook][0]  # [seq, n_heads, d_head]
        v = cache[v_hook][0]

        # Per-position, per-head L2 norms
        k_norms = k.norm(dim=-1)  # [seq, n_heads]
        v_norms = v.norm(dim=-1)

        # Mean across heads
        k_mean_norm = k_norms.mean(dim=1)  # [seq]
        v_mean_norm = v_norms.mean(dim=1)

        # Sink detection: positions where K-norm > mean + 2*std
        k_mean = k_mean_norm.mean()
        k_std = k_mean_norm.std()
        threshold = k_mean + 2 * k_std
        sink_positions = (k_mean_norm > threshold).nonzero(as_tuple=True)[0].cpu().tolist()

        # Influence score: K-norm * V-norm product (proxy for cache contribution)
        influence = (k_mean_norm * v_mean_norm).cpu().numpy()
        influence = (influence / (influence.sum() + 1e-8)).tolist()

        # Memory for this layer: 2 tensors * seq * n_heads * d_head * bytes_per_elem
        seq_len, n_heads, d_head = k.shape
        layer_memory = 2 * seq_len * n_heads * d_head * 2  # FP16

        return {
            "layer": layer_idx,
            "k_norms": k_norms.cpu().numpy().T.tolist(),       # [n_heads, seq]
            "v_norms": v_norms.cpu().numpy().T.tolist(),
            "k_mean_norm": k_mean_norm.cpu().tolist(),
            "v_mean_norm": v_mean_norm.cpu().tolist(),
            "sink_positions": sink_positions,
            "influence_scores": influence,
            "memory_bytes": layer_memory,
        }

    # ── Full standalone analysis ────────────────────────────────────

    def analyze(self, prompt: str) -> dict:
        """Run full KV-cache analysis on a prompt."""
        model = self._mgr._model
        if model is None:
            raise RuntimeError("No model loaded")

        with self._lock, torch.no_grad():
            tokens = model.to_tokens(prompt, prepend_bos=True)
            token_strs = model.to_str_tokens(prompt, prepend_bos=True)
            _, cache = model.run_with_cache(tokens)

            n_layers = model.cfg.n_layers
            n_heads = model.cfg.n_heads
            d_head = model.cfg.d_head

            kv_layers = []
            for layer_idx in range(n_layers):
                ld = self.extract_kv_layer(cache, layer_idx)
                if ld:
                    kv_layers.append(ld)

        # Global sinks: positions that are sinks in >50% of layers
        sink_counter = Counter()
        for ld in kv_layers:
            for pos in ld["sink_positions"]:
                sink_counter[pos] += 1
        threshold = len(kv_layers) * 0.5
        global_sinks = sorted(pos for pos, cnt in sink_counter.items()
                              if cnt >= threshold)

        total_mem = sum(ld["memory_bytes"] for ld in kv_layers)

        return {
            "tokens": list(token_strs),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_head": d_head,
            "layers": kv_layers,
            "total_memory_bytes": total_mem,
            "global_sink_positions": global_sinks,
            "model_name": self._mgr.current_model or "unknown",
        }

    # ── Adversarial: Prefix Poisoning ───────────────────────────────

    def prefix_poisoning(self, clean_prompt: str, poisoned_prompt: str,
                         progress_callback=None) -> dict:
        """Compare KV representations between clean and adversarial prefixes.

        Measures per-layer divergence in K/V norm distributions to show how
        an adversarial prefix shifts the cached representations that all
        subsequent tokens attend to.
        """
        clean_result = self.analyze(clean_prompt)
        if progress_callback:
            progress_callback({"phase": "clean", "progress": 0.5})

        poisoned_result = self.analyze(poisoned_prompt)
        if progress_callback:
            progress_callback({"phase": "poisoned", "progress": 1.0})

        divergence = []
        for cl, pl in zip(clean_result["layers"], poisoned_result["layers"]):
            min_len = min(len(cl["k_mean_norm"]), len(pl["k_mean_norm"]))
            k_diff = [abs(cl["k_mean_norm"][i] - pl["k_mean_norm"][i])
                      for i in range(min_len)]
            v_diff = [abs(cl["v_mean_norm"][i] - pl["v_mean_norm"][i])
                      for i in range(min_len)]
            divergence.append({
                "layer": cl["layer"],
                "k_divergence": k_diff,
                "v_divergence": v_diff,
                "mean_k_shift": sum(k_diff) / (len(k_diff) or 1),
                "mean_v_shift": sum(v_diff) / (len(v_diff) or 1),
            })

        return {
            "clean": clean_result,
            "poisoned": poisoned_result,
            "divergence": divergence,
        }

    # ── Adversarial: Cache Exhaustion ───────────────────────────────

    def cache_exhaustion(self, prompt: str,
                         context_lengths: list[int] | None = None) -> dict:
        """Project KV-cache memory usage to various context lengths.

        Demonstrates how an adversary can force massive memory allocation
        by keeping sessions alive with long or repeated inputs.
        """
        result = self.analyze(prompt)
        seq_len = len(result["tokens"])
        per_token_bytes = result["total_memory_bytes"] / seq_len if seq_len else 0

        if context_lengths is None:
            context_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

        projections = []
        for ctx_len in context_lengths:
            projected = per_token_bytes * ctx_len
            projections.append({
                "context_length": ctx_len,
                "projected_bytes": int(projected),
                "projected_mb": round(projected / (1024 * 1024), 2),
                "projected_gb": round(projected / (1024**3), 4),
            })

        return {
            "base_analysis": result,
            "per_token_bytes": per_token_bytes,
            "projections": projections,
        }

    # ── Adversarial: Quantization Drift ─────────────────────────────

    def quantization_drift(self, prompt: str,
                           progress_callback=None) -> dict:
        """Compare FP32 K/V norms against simulated FP16 and INT8 quantization.

        Shows where precision loss is maximal — positions an adversary can
        craft inputs to exploit, since post-quantization the K/V shift in a
        direction the attacker controls.
        """
        model = self._mgr._model
        if model is None:
            raise RuntimeError("No model loaded")

        with self._lock, torch.no_grad():
            tokens = model.to_tokens(prompt, prepend_bos=True)
            token_strs = model.to_str_tokens(prompt, prepend_bos=True)
            _, cache = model.run_with_cache(tokens)

            n_layers = model.cfg.n_layers
            drift_layers = []

            for layer_idx in range(n_layers):
                k_hook = f"blocks.{layer_idx}.attn.hook_k"
                v_hook = f"blocks.{layer_idx}.attn.hook_v"
                if k_hook not in cache:
                    continue

                k = cache[k_hook][0]  # [seq, n_heads, d_head]
                v = cache[v_hook][0]

                # FP32 ground truth norms
                k_fp32_norms = k.float().norm(dim=-1).mean(dim=1)  # [seq]
                v_fp32_norms = v.float().norm(dim=-1).mean(dim=1)

                # FP16 simulation
                k_fp16_norms = k.half().float().norm(dim=-1).mean(dim=1)
                v_fp16_norms = v.half().float().norm(dim=-1).mean(dim=1)

                # INT8 simulation (quantize/dequantize)
                k_scale = k.abs().max() / 127.0
                k_int8 = (k / (k_scale + 1e-8)).round().clamp(-128, 127) * k_scale
                k_int8_norms = k_int8.norm(dim=-1).mean(dim=1)

                v_scale = v.abs().max() / 127.0
                v_int8 = (v / (v_scale + 1e-8)).round().clamp(-128, 127) * v_scale
                v_int8_norms = v_int8.norm(dim=-1).mean(dim=1)

                k_fp16_drift = (k_fp32_norms - k_fp16_norms).abs().cpu().tolist()
                k_int8_drift = (k_fp32_norms - k_int8_norms).abs().cpu().tolist()
                v_fp16_drift = (v_fp32_norms - v_fp16_norms).abs().cpu().tolist()
                v_int8_drift = (v_fp32_norms - v_int8_norms).abs().cpu().tolist()

                drift_layers.append({
                    "layer": layer_idx,
                    "k_fp32_norms": k_fp32_norms.cpu().tolist(),
                    "v_fp32_norms": v_fp32_norms.cpu().tolist(),
                    "k_fp16_drift": k_fp16_drift,
                    "k_int8_drift": k_int8_drift,
                    "v_fp16_drift": v_fp16_drift,
                    "v_int8_drift": v_int8_drift,
                    "mean_k_fp16_drift": float(np.mean(k_fp16_drift)),
                    "mean_k_int8_drift": float(np.mean(k_int8_drift)),
                    "mean_v_fp16_drift": float(np.mean(v_fp16_drift)),
                    "mean_v_int8_drift": float(np.mean(v_int8_drift)),
                })

                if progress_callback:
                    progress_callback({
                        "progress": (layer_idx + 1) / n_layers,
                        "layer": layer_idx,
                    })

        return {
            "tokens": list(token_strs),
            "n_layers": n_layers,
            "layers": drift_layers,
            "model_name": self._mgr.current_model or "unknown",
        }

    # ── Adversarial: Cross-Turn Forensics ───────────────────────────

    def cross_turn_forensics(self, turns: list[str],
                             progress_callback=None) -> dict:
        """Simulate multi-turn conversation and track KV-cache state evolution.

        Shows how adversarial signal accumulates across conversation turns
        as cached K/V tensors persist and grow.
        """
        results = []
        accumulated = ""
        for i, turn in enumerate(turns):
            accumulated += (" " if accumulated else "") + turn
            analysis = self.analyze(accumulated)
            results.append({
                "turn": i,
                "turn_text": turn,
                "full_prompt_length": len(accumulated),
                "n_tokens": len(analysis["tokens"]),
                "total_memory_bytes": analysis["total_memory_bytes"],
                "global_sinks": analysis["global_sink_positions"],
                "layers_summary": [
                    {
                        "layer": ld["layer"],
                        "mean_k_norm": float(np.mean(ld["k_mean_norm"])),
                        "mean_v_norm": float(np.mean(ld["v_mean_norm"])),
                        "n_sinks": len(ld["sink_positions"]),
                    }
                    for ld in analysis["layers"]
                ],
            })
            if progress_callback:
                progress_callback({
                    "turn": i,
                    "total": len(turns),
                    "progress": (i + 1) / len(turns),
                })

        return {
            "turns": results,
            "model_name": self._mgr.current_model or "unknown",
        }

    def cancel(self):
        self._cancel.set()
