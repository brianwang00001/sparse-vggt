from functools import partial
from types import MethodType
from typing import Literal

import torch
from vggt.models.aggregator import Aggregator, slice_expand_and_flatten

from sparse_vggt.models.attention import adaptive_sparse_attention_forward
from sparse_vggt.models.utils import print_sparse_info
from sparse_vggt.utils.hilbert import hilbert_permute
from sparse_vggt.utils.sparse_wrapper import check_sparse_mode


def sparse_vggt_aggregator_forward(
    self,
    images: torch.Tensor,
    use_hilbert: bool = True,
    intermediate_layer_idx: list[int] = [4, 11, 17, 23],  # hardcoded for vggt
):
    """Adaptive Block Sparse Attention Aggregator forward pass to replace the original aggregator forward function.

    Args:
        images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
        use_hilbert: if True, use Hilbert permutation for patch tokens

    Returns:
        (list[torch.Tensor], int):
            The list of outputs from the attention blocks,
            and the patch_start_idx indicating where patch tokens begin.
    """
    B, N, C_in, H, W = images.shape

    if C_in != 3:
        raise ValueError(f"Expected 3 input channels, got {C_in}")

    # Normalize images and reshape for patch embed
    images = (images - self._resnet_mean) / self._resnet_std

    # Reshape to [B*N, C, H, W] for patch embedding
    images = images.view(B * N, C_in, H, W)
    patch_tokens = self.patch_embed(images)

    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]

    _, P, C = patch_tokens.shape

    # Expand camera and register tokens to match batch size and sequence length
    camera_token = slice_expand_and_flatten(self.camera_token, B, N)
    register_token = slice_expand_and_flatten(self.register_token, B, N)

    # Concatenate special tokens with patch tokens
    tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

    # convert to patch dimension
    H = H // self.patch_size
    W = W // self.patch_size
    pos = None
    if self.rope is not None:
        pos = self.position_getter(B * N, H, W, device=images.device)

    if self.patch_start_idx > 0 and pos is not None:
        # do not use position embedding for special tokens (camera and register tokens)
        # so set pos to 0 for the special tokens
        pos = pos + 1
        pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)

    # update P because we added special tokens
    _, P, C = tokens.shape

    frame_idx = 0
    global_idx = 0

    output_list = []

    # add: hilbert permutation
    S = self.patch_start_idx
    if use_hilbert:
        tokens = hilbert_permute(tokens, H, W, S)  # (B * N, H * W + S, C)
        pos = hilbert_permute(pos, H, W, S)  # (B * N, H * W, 2)

    concat_inter = None
    frame_intermediates = []
    global_intermediates = []

    for layer_idx in range(self.aa_block_num):
        for attn_type in self.aa_order:
            if attn_type == "frame":
                tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                    tokens, B, N, P, C, frame_idx, pos=pos
                )
            elif attn_type == "global":
                tokens, global_idx, global_intermediates = self._process_global_attention(
                    tokens, B, N, P, C, global_idx, pos=pos
                )
            else:
                raise ValueError(f"Unknown attention type: {attn_type}")
        if layer_idx in intermediate_layer_idx:
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x N x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
        else:
            output_list.append(None)  # workaround for vggt's heads

    del concat_inter
    del frame_intermediates
    del global_intermediates

    if use_hilbert:
        for i, inter_x in enumerate(output_list):
            # (B, N, H * W + S, C)
            if inter_x is not None:
                output_list[i] = hilbert_permute(inter_x, H, W, S, reverse=True)

    return output_list, self.patch_start_idx


def sparse_aggregator_from_vggt(
    aggregator: Aggregator,
    use_hilbert: bool = False,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    pool_mode: Literal["max", "avg"] = "avg",
    aux_output: bool = False,
    aux_sparsity_only: bool = True,
    num_special_tokens: int = 5,
    verbose: bool = True,
):
    """Convert the original VGGT aggregator to a sparse aggregator.

    Args:
        aggregator: Original VGGT aggregator.
        use_hilbert: If True, use Hilbert permutation on patch tokens.
        sparse_ratio: Sparse ratio for the global attention.
        cdf_threshold: CDF threshold for the global attention.
        pool_mode: Avg or Max pooling for the global attention.
        aux_output: If True, store auxiliary output from the global attention.
        aux_sparsity_only: If True, only store sparsity from the global attention.

    Returns:
        Aggregator: Modified Aggregator
        aux_output_store (dict | None): Auxiliary output storage.
    """
    # Check sparse mode
    check_sparse_mode(sparse_ratio, cdf_threshold)

    # Replace aggregator forward function
    aggregator_fwd = partial(sparse_vggt_aggregator_forward, use_hilbert=use_hilbert)
    aggregator.forward = MethodType(aggregator_fwd, aggregator)

    modified_layers = [i for i in range(24)]
    if aux_output:
        # Pointers to store auxiliary output
        aux_output_store = {i: {} for i in modified_layers}
    else:
        # No auxiliary output
        aux_output_store = {i: None for i in modified_layers}

    for i in range(len(aggregator.global_blocks)):
        if i not in modified_layers:
            continue

        # Replace attention forward function
        attn_fwd = partial(
            adaptive_sparse_attention_forward,
            sparse_ratio=sparse_ratio,
            cdf_threshold=cdf_threshold,
            pool_mode=pool_mode,
            aux_output_store=aux_output_store[i],
            aux_sparsity_only=aux_sparsity_only,
            num_special_tokens=num_special_tokens,
        )
        aggregator.global_blocks[i].attn.forward = MethodType(
            attn_fwd, aggregator.global_blocks[i].attn
        )

    if verbose:
        print_sparse_info(
            sparse_ratio,
            cdf_threshold,
            pool_mode,
            use_hilbert,
            aux_output,
            aux_sparsity_only,
        )
    return aggregator, aux_output_store
