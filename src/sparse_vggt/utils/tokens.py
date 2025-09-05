def get_patch_tokens(x, N, P, S):
    """Get patch tokens from the input tensor.

    Args:
        x (torch.Tensor): input tensor of shape (..., N * P, C)

    Returns:
        patch (torch.Tensor): patch tokens of shape (..., N * (P - S), C)

    where
        N: number of frames
        P: number of patch tokens + special tokens
        S: number of special tokens
    """
    assert x.shape[-2] == N * P

    batch_dims = x.shape[:-2]
    channels = x.shape[-1]

    x = x.view(batch_dims + (N, P, channels))
    patch = x[..., S:, :]  # (..., N, P - S, C)
    patch = patch.reshape(batch_dims + (N * (P - S), channels))
    return patch.contiguous()


def get_special_tokens(x, N, P, S):
    """Get special tokens from the input tensor.

    Args:
        x (torch.Tensor): input tensor of shape (..., N * P, C)

    Returns:
        special (torch.Tensor): special tokens of shape (..., N * S, C)
    """
    assert x.shape[-2] == N * P

    batch_dims = x.shape[:-2]
    channels = x.shape[-1]

    x = x.view(batch_dims + (N, P, channels))
    special = x[..., :S, :]  # (..., N, S, C)
    special = special.reshape(batch_dims + (N * S, channels))
    if special.numel() == 0:
        return None
    return special
