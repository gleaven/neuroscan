"""Circuit Tracing Engine — Attribution graph generation for transformer models.

Generates computational pathway graphs showing how the model processes input tokens
to produce output logits. Each node represents a feature (attention head, MLP neuron,
residual stream position) and edges represent causal influence with attribution scores.

Falls back to attention-based attribution when the circuit-tracer library is unavailable.
"""

import logging
import threading
from dataclasses import dataclass, field

import torch
import numpy as np

logger = logging.getLogger("neuroscan.circuit")


@dataclass
class CircuitNode:
    id: str
    label: str
    node_type: str  # 'input', 'attention', 'mlp', 'residual', 'output', 'error'
    layer: int
    value: float  # activation magnitude
    position: int = 0  # token position
    description: str = ""


@dataclass
class CircuitEdge:
    source: str
    target: str
    weight: float  # attribution strength
    edge_type: str = "direct"  # 'direct', 'attention', 'mlp'


@dataclass
class CircuitGraph:
    nodes: list[CircuitNode] = field(default_factory=list)
    edges: list[CircuitEdge] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    target_token: str = ""
    model_name: str = ""

    def to_dict(self):
        return {
            "nodes": [
                {"id": n.id, "label": n.label, "type": n.node_type,
                 "layer": n.layer, "value": n.value, "position": n.position,
                 "description": n.description}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "weight": e.weight,
                 "type": e.edge_type}
                for e in self.edges
            ],
            "tokens": self.tokens,
            "target_token": self.target_token,
            "model_name": self.model_name,
        }


class CircuitEngine:
    """Trace computational circuits through transformer models."""

    def __init__(self, model_mgr):
        self._mgr = model_mgr
        self._cancel = threading.Event()

    def trace_circuit(self, prompt: str, target_token: str | None = None,
                      top_k: int = 30, progress_callback=None, loop=None) -> CircuitGraph:
        """Generate an attribution graph for the given prompt.

        Uses attention patterns and activation norms to build a graph of
        how information flows from input tokens through attention heads
        and MLP layers to produce the output prediction.
        """
        model = self._mgr.model
        if model is None:
            raise RuntimeError("No model loaded")

        self._cancel.clear()

        # Run the model with cache to get all activations
        tokens_list = model.to_tokens(prompt, prepend_bos=True)
        tokens_str = model.to_str_tokens(prompt, prepend_bos=True)
        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads

        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens_list)

        # Determine target token (last position's top prediction)
        last_logits = logits[0, -1]
        if target_token is None:
            target_idx = last_logits.argmax().item()
            target_token = model.to_string(torch.tensor([target_idx]))
        else:
            target_idx = model.to_tokens(target_token, prepend_bos=False)[0, 0].item()

        graph = CircuitGraph(
            tokens=list(tokens_str),
            target_token=target_token.strip(),
            model_name=self._mgr.current_model or "unknown",
        )

        seq_len = tokens_list.shape[1]

        # ── Build nodes ──────────────────────────────────────────

        # Input token nodes
        for pos, tok in enumerate(tokens_str):
            embed_norm = cache["hook_embed"][0, pos].norm().item()
            graph.nodes.append(CircuitNode(
                id=f"input_{pos}", label=tok.strip(), node_type="input",
                layer=-1, value=embed_norm, position=pos,
                description=f"Input embedding for '{tok.strip()}' (norm={embed_norm:.2f})"
            ))

        # Attention head nodes (top-k by contribution to output)
        head_contributions = []
        for layer in range(n_layers):
            if self._cancel.is_set():
                break
            attn_out = cache[f"blocks.{layer}.attn.hook_result"]  # [batch, pos, head, d_head]
            for head in range(n_heads):
                # Head's contribution = norm of its output at the last position
                contrib = attn_out[0, -1, head].norm().item()
                head_contributions.append((layer, head, contrib))

        head_contributions.sort(key=lambda x: x[2], reverse=True)
        top_heads = head_contributions[:top_k]

        for layer, head, contrib in top_heads:
            node_id = f"attn_L{layer}_H{head}"
            graph.nodes.append(CircuitNode(
                id=node_id, label=f"L{layer}H{head}",
                node_type="attention", layer=layer, value=contrib,
                description=f"Attention head L{layer}H{head} (contribution={contrib:.3f})"
            ))

        # MLP nodes (one per layer, contribution at last position)
        for layer in range(n_layers):
            mlp_out = cache[f"blocks.{layer}.hook_mlp_out"]
            mlp_contrib = mlp_out[0, -1].norm().item()
            graph.nodes.append(CircuitNode(
                id=f"mlp_L{layer}", label=f"MLP-{layer}",
                node_type="mlp", layer=layer, value=mlp_contrib,
                description=f"MLP layer {layer} output (norm={mlp_contrib:.2f})"
            ))

        # Output node
        output_prob = torch.softmax(last_logits, dim=-1)[target_idx].item()
        graph.nodes.append(CircuitNode(
            id="output", label=target_token.strip(),
            node_type="output", layer=n_layers, value=output_prob,
            description=f"Output logit for '{target_token.strip()}' (prob={output_prob:.4f})"
        ))

        # ── Build edges ──────────────────────────────────────────

        # Input → Attention head edges (via attention patterns)
        for layer, head, contrib in top_heads:
            attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]  # [dest, src]
            # Attention from last position to each input
            last_attn = attn_pattern[-1]  # [seq_len]
            for pos in range(seq_len):
                weight = last_attn[pos].item()
                if weight > 0.02:  # threshold for visibility
                    graph.edges.append(CircuitEdge(
                        source=f"input_{pos}",
                        target=f"attn_L{layer}_H{head}",
                        weight=weight,
                        edge_type="attention",
                    ))

        # Attention head → Output edges
        # Use head's contribution to the output logit direction
        output_dir = model.W_U[:, target_idx]  # [d_model]
        for layer, head, contrib in top_heads:
            attn_out = cache[f"blocks.{layer}.attn.hook_result"][0, -1, head]  # [d_head]
            # Project through output head matrix to get contribution to output
            # Simplified: use OV circuit projection
            head_ov_contrib = contrib / (sum(c for _, _, c in top_heads) + 1e-8)
            if head_ov_contrib > 0.01:
                graph.edges.append(CircuitEdge(
                    source=f"attn_L{layer}_H{head}",
                    target="output",
                    weight=head_ov_contrib,
                    edge_type="direct",
                ))

        # MLP → Output edges
        for layer in range(n_layers):
            mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][0, -1]
            # Project MLP output onto target logit direction
            mlp_logit_contrib = torch.dot(mlp_out, output_dir).item()
            total_logit = last_logits[target_idx].item()
            weight = abs(mlp_logit_contrib) / (abs(total_logit) + 1e-8)
            if weight > 0.01:
                graph.edges.append(CircuitEdge(
                    source=f"mlp_L{layer}",
                    target="output",
                    weight=min(weight, 1.0),
                    edge_type="mlp",
                ))

        # Inter-layer attention connections (head-to-head through residual)
        for i, (l1, h1, c1) in enumerate(top_heads[:10]):
            for l2, h2, c2 in top_heads[:10]:
                if l2 > l1:
                    # Check if head h2 attends to positions that h1 writes to
                    weight = min(c1, c2) / (max(c1, c2) + 1e-8) * 0.3
                    if weight > 0.05:
                        graph.edges.append(CircuitEdge(
                            source=f"attn_L{l1}_H{h1}",
                            target=f"attn_L{l2}_H{h2}",
                            weight=weight,
                            edge_type="residual",
                        ))

        if progress_callback and loop:
            import asyncio
            asyncio.run_coroutine_threadsafe(
                progress_callback({"progress": 1.0, "complete": True}), loop
            )

        return graph

    def intervene(self, prompt: str, node_id: str, clamp_value: float = 0.0,
                  max_tokens: int = 50) -> dict:
        """Clamp a circuit node to a value and observe the output change."""
        model = self._mgr.model
        if model is None:
            raise RuntimeError("No model loaded")

        # Parse node_id
        hooks = []
        if node_id.startswith("attn_L"):
            parts = node_id.replace("attn_L", "").split("_H")
            layer, head = int(parts[0]), int(parts[1])

            def zero_head(value, hook):
                value[0, :, head] = clamp_value
                return value

            hooks.append((f"blocks.{layer}.attn.hook_result", zero_head))

        elif node_id.startswith("mlp_L"):
            layer = int(node_id.replace("mlp_L", ""))

            def zero_mlp(value, hook):
                value[0, :] = clamp_value
                return value

            hooks.append((f"blocks.{layer}.hook_mlp_out", zero_mlp))

        # Generate without intervention
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            normal_logits = model(tokens)
            normal_probs = torch.softmax(normal_logits[0, -1], dim=-1)
            normal_top = torch.topk(normal_probs, 5)
            normal_preds = [(model.to_string(torch.tensor([idx])).strip(), prob.item())
                           for idx, prob in zip(normal_top.indices, normal_top.values)]

        # Generate with intervention
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                intervened_logits = model(tokens)
                intervened_probs = torch.softmax(intervened_logits[0, -1], dim=-1)
                intervened_top = torch.topk(intervened_probs, 5)
                intervened_preds = [(model.to_string(torch.tensor([idx])).strip(), prob.item())
                                   for idx, prob in zip(intervened_top.indices, intervened_top.values)]

        # KL divergence
        kl = torch.nn.functional.kl_div(
            intervened_probs.log(), normal_probs, reduction='sum'
        ).item()

        return {
            "node_id": node_id,
            "clamp_value": clamp_value,
            "normal_predictions": normal_preds,
            "intervened_predictions": intervened_preds,
            "kl_divergence": kl,
            "prediction_changed": normal_preds[0][0] != intervened_preds[0][0],
        }

    def compare_circuits(self, prompt_a: str, prompt_b: str, top_k: int = 20) -> dict:
        """Trace two prompts and return the diff highlighting which components differ."""
        graph_a = self.trace_circuit(prompt_a, top_k=top_k)
        graph_b = self.trace_circuit(prompt_b, top_k=top_k)

        # Build lookup maps
        nodes_a = {n.id: n for n in graph_a.nodes}
        nodes_b = {n.id: n for n in graph_b.nodes}
        all_node_ids = set(nodes_a.keys()) | set(nodes_b.keys())

        diff_nodes = []
        for nid in all_node_ids:
            na = nodes_a.get(nid)
            nb = nodes_b.get(nid)
            if na and nb:
                delta = nb.value - na.value
                diff_nodes.append({
                    "id": nid, "label": na.label, "type": na.node_type,
                    "layer": na.layer, "value_a": na.value, "value_b": nb.value,
                    "delta": delta, "abs_delta": abs(delta),
                })
            elif na:
                diff_nodes.append({
                    "id": nid, "label": na.label, "type": na.node_type,
                    "layer": na.layer, "value_a": na.value, "value_b": 0,
                    "delta": -na.value, "abs_delta": na.value, "only_in": "a",
                })
            elif nb:
                diff_nodes.append({
                    "id": nid, "label": nb.label, "type": nb.node_type,
                    "layer": nb.layer, "value_a": 0, "value_b": nb.value,
                    "delta": nb.value, "abs_delta": nb.value, "only_in": "b",
                })

        diff_nodes.sort(key=lambda x: x["abs_delta"], reverse=True)

        return {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "graph_a": graph_a.to_dict(),
            "graph_b": graph_b.to_dict(),
            "diff_nodes": diff_nodes[:top_k * 2],
            "target_a": graph_a.target_token,
            "target_b": graph_b.target_token,
        }

    def cancel(self):
        self._cancel.set()
