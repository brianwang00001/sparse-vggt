import math

import torch
import torch.nn.functional as F
from einops import rearrange

from sparse_vggt.utils.sparse_wrapper import block_sparse_attn_cuda
from sparse_vggt.utils.tokens import get_patch_tokens, get_special_tokens


def predict_attention(query, key, ks_q=128, ks_k=64, pool_mode="avg"):
    """
    Args:
        query: (B, nh, Tq, C)
        key: (B, nh, Tk, C)

    Return:
        pooled_prob: (B, nh, Tq, Tk)
    """
    assert pool_mode in ["max", "avg"], f"{pool_mode=}"

    pooling_fn = {
        "max": F.max_pool1d,
        "avg": F.avg_pool1d,
    }[pool_mode]

    assert query.ndim == 4, f"{query.shape=}"
    assert key.ndim == 4, f"{key.shape=}"

    B, nh, Tq, C = query.shape
    _, _, Tk, _ = key.shape

    # Query Pooling
    query = rearrange(query, "B nh Tq C -> (B nh) C Tq")
    pooled_query = pooling_fn(query, kernel_size=ks_q, ceil_mode=True)
    pooled_query = rearrange(pooled_query, "(B nh) C Tq -> B nh Tq C", B=B, nh=nh)

    # Key Pooling
    key = rearrange(key, "B nh Tk C -> (B nh) C Tk")
    pooled_key = pooling_fn(key, kernel_size=ks_k, ceil_mode=True)
    pooled_key = rearrange(pooled_key, "(B nh) C Tk -> B nh Tk C", B=B, nh=nh)

    # Dot Product
    scale = 1 / math.sqrt(C)
    pooled_score = pooled_query @ pooled_key.transpose(-1, -2) * scale  # (B, nh, Tq, Tk)
    pooled_prob = F.softmax(pooled_score, dim=-1)

    return pooled_prob


def adaptive_sparse_attention_forward(
    self,
    x,
    pos,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    pool_mode: str = "avg",
    aux_sparsity_only: bool | None = None,
    aux_output_store: dict | None = None,
    num_special_tokens: int = 5,
    num_heads: int = 16,
):
    """Adaptive Block Sparse Attention forward pass to replace the original attention forward function.

    Args:
        x: (B, N * P, hidden_dim) == (B, N * (H * W + S), hidden_dim)

    Return:
        x: (B, N * P, hidden_dim)
    """

    B, NP, hidden_dim = x.shape
    S = num_special_tokens
    nh = num_heads
    hd = hidden_dim // num_heads

    # Infer H, W and N from pos
    H = int(pos[0].max(0).values[0])
    W = int(pos[0].max(0).values[1])
    N = NP // (H * W + S)
    P = H * W + S

    # Sanity check
    assert N * (H * W + S) == NP, f"{N=}, {H=}, {W=}, {S=}, {NP=}"

    qkv = self.qkv(x)
    three = 3
    qkv = rearrange(qkv, "B N (three nh hd)-> B N three nh hd", three=three, nh=nh, hd=hd)
    qkv = rearrange(qkv, "B N three nh hd -> three B nh N hd")

    q, k, v = qkv.unbind(0)  # (B, num_heads, N * P, head_dim)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.rope is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)

    # separate patch and special tokens
    q_special = get_special_tokens(q, N, P, S)
    k_special = get_special_tokens(k, N, P, S)
    v_special = get_special_tokens(v, N, P, S)
    q_patch = get_patch_tokens(q, N, P, S)
    k_patch = get_patch_tokens(k, N, P, S)
    v_patch = get_patch_tokens(v, N, P, S)

    # special tokens attend to all tokens
    if q_special is not None:
        x_special = F.scaled_dot_product_attention(q_special, k, v)
    else:
        x_special = None
    # release memory
    del q, k, v, qkv

    # Append special key and values in the end
    if k_special is not None:
        key = torch.cat([k_patch, k_special], dim=-2)
        value = torch.cat([v_patch, v_special], dim=-2)
    else:
        key = k_patch
        value = v_patch

    if self.training:
        raise NotImplementedError("This is currently only training-free. Use .eval()")

    else:
        attn_pooled = predict_attention(query=q_patch, key=k_patch, pool_mode=pool_mode)
        # release memory
        del k_patch, v_patch

        # patch attention
        x_patch, sparsity = block_sparse_attn_cuda(
            query=q_patch,
            key=key,
            value=value,
            pooled_score=attn_pooled,
            sparse_ratio=sparse_ratio,
            cdf_threshold=cdf_threshold,
            return_sparsity=True,
        )

    if aux_output_store is not None:
        aux_output_store["sparsity"] = sparsity

        if not aux_sparsity_only:
            aux_output_store.update(
                {
                    "attn_pooled": attn_pooled,
                    "query": q,
                    "key": k,
                    "shape": {
                        "B": B,
                        "N": N,
                        "P": P,
                        "head_dim": hd,
                        "num_heads": nh,
                        "H": H,
                        "W": W,
                    },
                }
            )

    x_patch = rearrange(
        x_patch,
        "B nh (N H W) hd -> B nh N (H W) hd",
        N=N,
        H=H,
        W=W,
    )

    # combine patch and special tokens
    if x_special is not None:
        x = x_special.view(B, nh, N, S, hd)
        x = torch.cat([x, x_patch], dim=-2)
    else:
        x = x_patch

    x = x.view(B, nh, N * P, hd)
    x = x.transpose(1, 2).reshape(B, N * P, nh * hd)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
