import math

import spas_sage_attn._qattn as qattn
import torch
from spas_sage_attn.quant_per_block import per_block_int8
from spas_sage_attn.utils import (
    block_map_lut_triton,
    fill_block_map_triton,
    hyperparameter_check,
)


def check_sparse_mode(sparse_ratio, cdf_threshold):
    """Check the valid combinations of sparse_ratio, and cdf_threshold for sparse inference.

    Args:
        sparse_ratio (float | None): choose a ratio of top blocks
        cdf_threshold (float | None): choose blocks that accumulate to a certain threshold

    Four modes (combinations) are allowed:
        1. only specify sparse_ratio
        2. only specify cdf_threshold
        3. specify both sparse_ratio and cdf_threshold
            This means that the cdf threshold and sparse ratio are BOTH reached.
    """
    use_ratio = sparse_ratio is not None
    use_cdf = cdf_threshold is not None

    # Modes
    only_use_ratio = use_ratio and (not use_cdf)
    only_use_cdf = (not use_ratio) and use_cdf
    use_ratio_and_cdf = use_ratio and use_cdf

    assert (
        only_use_ratio + only_use_cdf + use_ratio_and_cdf == 1
    ), f"Current: {sparse_ratio=}, {cdf_threshold=}"


def check_sparse_mode_three_type(topk, sparse_ratio, cdf_threshold):
    """Check the valid combinations of topk, sparse_ratio, and cdf_threshold for sparse inference.

    Args:
        topk (int | None): choose the top-k key blocks for each query block
        sparse_ratio (float | None): choose a ratio of top blocks
        cdf_threshold (float | None): choose blocks that accumulate to a certain threshold

    Four modes (combinations) are allowed:
        1. only specify topk
        2. only specify sparse_ratio
        3. only specify cdf_threshold
        4. specify both sparse_ratio and cdf_threshold
            This means that the cdf threshold and sparse ratio are BOTH reached.
    """
    use_topk = topk is not None
    use_ratio = sparse_ratio is not None
    use_cdf = cdf_threshold is not None

    # Modes
    only_use_topk = use_topk and (not use_ratio) and (not use_cdf)
    only_use_ratio = (not use_topk) and use_ratio and (not use_cdf)
    only_use_cdf = (not use_topk) and (not use_ratio) and use_cdf
    use_ratio_and_cdf = use_ratio and use_cdf and (not use_topk)

    assert (
        only_use_topk + only_use_ratio + only_use_cdf + use_ratio_and_cdf == 1
    ), f"Current: {topk=}, {sparse_ratio=}, {cdf_threshold=}"


def get_block_mask(
    pooled_score: torch.Tensor,
    sink_blocks: int,
    topk: int | None = None,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Args:
        pooled_score (Tensor): Pooled attention scores after softmax
            Shape: (B, nh, q_blk, k_blk)
            where q_blk and k_blk are the number of query and key blocks.
            nh: number of heads
        sink_blocks (int): number of key blocks (from the beginning) to always be selected

    Returns:
        final_map (Bool Tensor): (B, nh, q_blk, k_blk + sink_blocks)
            True means the block is selected.

    """
    check_sparse_mode_three_type(topk, sparse_ratio, cdf_threshold)

    B, nh, q_blk, k_blk = pooled_score.shape
    assert sink_blocks >= 0 and sink_blocks <= k_blk

    if sparse_ratio is not None:
        # Convert sparse ratio to topk
        assert sparse_ratio >= 0 and sparse_ratio <= 1
        topk = int(k_blk * (1 - sparse_ratio))

    if topk is not None:
        assert topk >= 0 and topk <= k_blk

    if cdf_threshold is not None:
        assert cdf_threshold >= 0 and cdf_threshold <= 1

    sorted_score = torch.sort(pooled_score, dim=-1, descending=True)

    num_to_select = None
    if cdf_threshold is not None:
        cdf = torch.cumsum(sorted_score.values, dim=-1)
        cdfthreshd = hyperparameter_check(cdf_threshold, nh, pooled_score.device)
        cdfthreshd_ts = cdfthreshd.view(1, nh, 1, 1)
        cdfthreshd_ts = cdfthreshd_ts + eps  # to avoid numerical error in searchsorted
        cdfthreshd_ts = cdfthreshd_ts.expand(B, -1, q_blk, 1).contiguous()
        num_to_select = torch.searchsorted(cdf, cdfthreshd_ts, right=True).squeeze(-1)

    if topk is not None:
        if num_to_select is None:
            num_to_select = torch.full((B, nh, q_blk), topk, device=pooled_score.device)
        else:
            num_to_select = torch.clamp(num_to_select, min=topk)

    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)

    if sink_blocks > 0:
        # Always select special tokens/blocks
        ones_shape = list(final_map.shape)
        ones_shape[-1] = sink_blocks
        trailing_ones = torch.ones(ones_shape, device=final_map.device).bool()
        final_map = torch.cat([final_map, trailing_ones], dim=-1)

    return final_map


def block_sparse_attn_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pooled_score: torch.Tensor,
    topk: int | None = None,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    return_sparsity: bool = False,
    dtype: torch.dtype = torch.float16,
    out_dtype: torch.dtype = torch.float32,
):
    """Block sparse attention using SpargeAttn kernels

    Args:
        query (torch.Tensor): (B, nheads, Tq, head_dim)
        key (torch.Tensor): (B, nheads, Tk, head_dim)
        value (torch.Tensor): (B, nheads, Tk, head_dim)
            sink tokens are appended to the end of key and value.
        pooled_score (torch.Tensor): (B, nheads, q_blk, k_blk)
            where q_blk and k_blk are the number of query and key blocks.
            The score here *doesn't* contain the sink tokens.
        topk, sparse_ratio, cdf_threshold: the mode of sparse attention
            - topk: choose the top-k key blocks for each query block
            - sparse_ratio: choose a ratio of top blocks
            - cdf_threshold: choose blocks that accumulate to a certain threshold

    Returns:
        out: Attention output of shape (B, nheads, T, head_dim)
    """
    # Hardcode some arguments for using SpargeAttn kernels
    _is_causal = 0
    KBLK = 64
    pvthreshd = 1e10
    pvthreshd = hyperparameter_check(pvthreshd, query.size(-3), query.device)

    # Get block mask
    Tk = key.shape[-2]
    orig_Kblk = pooled_score.shape[-1]
    total_Kblk = math.ceil(Tk / KBLK)
    sink_blocks = total_Kblk - orig_Kblk
    final_map = get_block_mask(
        pooled_score,
        sink_blocks=sink_blocks,
        topk=topk,
        sparse_ratio=sparse_ratio,
        cdf_threshold=cdf_threshold,
    )
    lut, valid_block_num = block_map_lut_triton(final_map)

    # Type conversion
    query, key, value = (
        query.contiguous().to(dtype),
        key.contiguous().to(dtype),
        value.contiguous().to(dtype),
    )

    # Quantization
    km = key.mean(dim=-2, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = per_block_int8(query, key - km)
    q_scale = q_scale.squeeze(-1)
    k_scale = k_scale.squeeze(-1)

    # Get softmax scale
    hd = query.shape[-1]
    scale = 1.0 / (hd**0.5)

    # SpargeAttn attention kernel
    o = torch.empty_like(query)
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
        q_int8,
        k_int8,
        value,
        o,
        lut,
        valid_block_num,
        pvthreshd,
        q_scale,
        k_scale,
        1,
        _is_causal,
        1,
        scale,
        0,
    )
    o = o.to(out_dtype)
    if return_sparsity:
        sparsity = 1 - final_map.float().mean().item()
        return o, sparsity
    else:
        return o
