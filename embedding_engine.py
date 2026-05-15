"""Embedding Security Engine — Embedding inversion and privacy analysis.

Demonstrates that text can be partially reconstructed from embedding vectors,
exposing a critical privacy vulnerability in RAG systems and embedding-based
search. Also provides embedding space visualization via dimensionality reduction.
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger("neuroscan.embedding")


class EmbeddingEngine:
    """Analyze embedding security — inversion attacks and space visualization."""

    def __init__(self, model_mgr):
        self._mgr = model_mgr

    def embed_text(self, text: str) -> dict:
        """Embed text using the model's embedding layer.

        Returns the embedding vector and metadata about the representation.
        """
        model = self._mgr.model
        if model is None:
            raise RuntimeError("No model loaded")

        tokens = model.to_tokens(text, prepend_bos=True)
        tokens_str = model.to_str_tokens(text, prepend_bos=True)

        with torch.no_grad():
            # Get embeddings from the model's embedding layer
            embeds = model.embed(tokens)  # [batch, seq, d_model]
            # Mean pooling for a single embedding vector
            embedding = embeds[0].mean(dim=0)  # [d_model]

        return {
            "text": text,
            "tokens": list(tokens_str),
            "n_tokens": len(tokens_str),
            "embedding": embedding.cpu().tolist(),
            "embedding_dim": embedding.shape[0],
            "embedding_norm": embedding.norm().item(),
        }

    def invert_embedding(self, embedding: list[float], n_steps: int = 100,
                          n_tokens: int = 10) -> dict:
        """Attempt to reconstruct text from an embedding vector.

        Uses iterative nearest-neighbor search in vocabulary space to find
        tokens whose embeddings best reconstruct the target vector.
        """
        model = self._mgr.model
        if model is None:
            raise RuntimeError("No model loaded")

        target = torch.tensor(embedding, device=model.cfg.device, dtype=torch.float32)
        target_norm = target / (target.norm() + 1e-8)

        # Get the full embedding matrix (vocab × d_model)
        W_E = model.W_E  # [vocab, d_model]

        # Normalize embeddings for cosine similarity
        W_E_norm = W_E / (W_E.norm(dim=-1, keepdim=True) + 1e-8)

        # Find top tokens by cosine similarity to target
        similarities = torch.matmul(W_E_norm, target_norm)  # [vocab]
        top_k = min(n_tokens * 3, 50)
        top_indices = torch.topk(similarities, top_k).indices

        # Greedy reconstruction: pick tokens that best reconstruct the embedding
        reconstructed_tokens = []
        remaining = target.clone()

        for step in range(n_tokens):
            # Find the token whose embedding best matches the remaining vector
            sims = torch.matmul(W_E_norm, remaining / (remaining.norm() + 1e-8))
            best_idx = sims.argmax().item()
            best_token = model.to_string(torch.tensor([best_idx])).strip()

            reconstructed_tokens.append({
                "token": best_token,
                "similarity": sims[best_idx].item(),
                "position": step,
            })

            # Subtract the matched embedding from the residual
            remaining = remaining - W_E[best_idx] * sims[best_idx].item()

        # Compute reconstruction quality
        reconstructed_text = " ".join(t["token"] for t in reconstructed_tokens)
        # Reconstruct embedding from chosen tokens
        chosen_embeds = torch.stack([
            W_E[model.to_tokens(t["token"], prepend_bos=False)[0, 0]]
            for t in reconstructed_tokens
        ])
        reconstructed_vec = chosen_embeds.mean(dim=0)
        cosine_sim = torch.cosine_similarity(target.unsqueeze(0), reconstructed_vec.unsqueeze(0)).item()

        return {
            "reconstructed_tokens": reconstructed_tokens,
            "reconstructed_text": reconstructed_text,
            "reconstruction_similarity": cosine_sim,
            "n_tokens_recovered": len(reconstructed_tokens),
            "residual_norm": remaining.norm().item(),
            "target_norm": target.norm().item(),
        }

    def analyze_embedding_space(self, texts: list[str], method: str = "pacmap") -> dict:
        """Project multiple text embeddings into 2D for visualization.

        Shows how texts cluster in embedding space — useful for understanding
        what the model considers "similar" and for detecting biases.
        """
        model = self._mgr.model
        if model is None:
            raise RuntimeError("No model loaded")

        embeddings = []
        tokens_list = []

        for text in texts:
            tokens = model.to_tokens(text, prepend_bos=True)
            tokens_str = model.to_str_tokens(text, prepend_bos=True)
            with torch.no_grad():
                embeds = model.embed(tokens)
                embedding = embeds[0].mean(dim=0).cpu().numpy()
            embeddings.append(embedding)
            tokens_list.append(list(tokens_str))

        embeddings_np = np.array(embeddings)

        # Dimensionality reduction
        if method == "pacmap" and len(texts) >= 5:
            try:
                import pacmap
                reducer = pacmap.PaCMAP(n_components=2, n_neighbors=min(10, len(texts) - 1))
                coords_2d = reducer.fit_transform(embeddings_np)
            except ImportError:
                from sklearn.decomposition import PCA
                coords_2d = PCA(n_components=2).fit_transform(embeddings_np)
        else:
            from sklearn.decomposition import PCA
            n_comp = min(2, embeddings_np.shape[0], embeddings_np.shape[1])
            coords_2d = PCA(n_components=n_comp).fit_transform(embeddings_np)
            if n_comp < 2:
                coords_2d = np.column_stack([coords_2d, np.zeros(len(texts))])

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8
        normalized = embeddings_np / norms
        similarity_matrix = (normalized @ normalized.T).tolist()

        points = []
        for i, text in enumerate(texts):
            points.append({
                "text": text[:80],
                "x": float(coords_2d[i, 0]),
                "y": float(coords_2d[i, 1]),
                "norm": float(norms[i, 0]),
            })

        return {
            "points": points,
            "similarity_matrix": similarity_matrix,
            "method": method,
            "n_texts": len(texts),
            "embedding_dim": embeddings_np.shape[1],
        }

    def privacy_attack(self, text: str, n_reconstruction_tokens: int = 15) -> dict:
        """Full privacy attack demo: embed → invert → compare.

        Shows side-by-side original vs reconstructed text with token matching.
        """
        # Step 1: Embed
        embed_result = self.embed_text(text)

        # Step 2: Invert
        invert_result = self.invert_embedding(
            embed_result["embedding"],
            n_tokens=n_reconstruction_tokens,
        )

        # Step 3: Compare
        original_tokens = set(t.strip().lower() for t in embed_result["tokens"])
        recovered_tokens = set(t["token"].strip().lower() for t in invert_result["reconstructed_tokens"])
        matching = original_tokens & recovered_tokens
        recovery_rate = len(matching) / max(len(original_tokens), 1)

        # Token-level comparison
        token_comparison = []
        for t in embed_result["tokens"]:
            t_clean = t.strip().lower()
            token_comparison.append({
                "token": t.strip(),
                "recovered": t_clean in recovered_tokens,
            })

        return {
            "original_text": text,
            "original_tokens": list(embed_result["tokens"]),
            "reconstructed_text": invert_result["reconstructed_text"],
            "reconstructed_tokens": invert_result["reconstructed_tokens"],
            "reconstruction_similarity": invert_result["reconstruction_similarity"],
            "token_recovery_rate": recovery_rate,
            "matching_tokens": list(matching),
            "token_comparison": token_comparison,
            "embedding_dim": embed_result["embedding_dim"],
        }
