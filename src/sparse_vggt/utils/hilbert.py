import math
from functools import lru_cache
from typing import List

import torch
from einops import rearrange, repeat


def gilbert2d(width, height):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
    of size (width x height).
    """

    if width >= height:
        yield from generate2d(0, 0, width, 0, 0, height)
    else:
        yield from generate2d(0, 0, 0, height, width, 0)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate2d(x, y, ax, ay, bx, by):
    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay))  # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by))  # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(0, w):
            yield (x, y)
            (x, y) = (x + dax, y + day)
        return

    if w == 1:
        # trivial column fill
        for i in range(0, h):
            yield (x, y)
            (x, y) = (x + dbx, y + dby)
        return

    (ax2, ay2) = (ax // 2, ay // 2)
    (bx2, by2) = (bx // 2, by // 2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
        yield from generate2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
        )


@lru_cache(maxsize=16)
def make_hilbert_gather_idx(width, height, inverse=False) -> List[int]:
    points = list(gilbert2d(width, height))
    mapping = {y * width + x: i for i, (x, y) in enumerate(points)}

    if inverse:
        mapping = {v: k for k, v in mapping.items()}

    gather_idx = [0] * (width * height)
    for old_idx, new_idx in mapping.items():
        gather_idx[new_idx] = old_idx
    return gather_idx


def get_hilbert_permutation(B, N, H, W, C, device, inverse=False):
    """Get the gather indices for the Hilbert permutation.

    Args:
        inverse (bool): If True, return the gather indices from original to Hilbert permutated.
            Otherwise, return the gather indices from Hilbert permutated to original.

    Returns:
        gather_idx: gather indices from original to Hilbert permutated
            shape: (B, N * H * W, C)
    """
    gather_idx = make_hilbert_gather_idx(W, H, inverse=inverse)
    gather_idx = torch.tensor(gather_idx, dtype=torch.int64, device=device)
    gather_idx = repeat(gather_idx, "T -> B T C", B=B, C=C)  # (B, H * W, C)

    # Applye to every frame
    frame_offset = torch.arange(N, device=device) * H * W
    frame_offset = rearrange(frame_offset, "N -> 1 N 1 1")
    gather_idx = repeat(gather_idx, "B T C -> B N T C", N=N)
    gather_idx = gather_idx + frame_offset

    gather_idx = rearrange(gather_idx, "B N T C -> B (N T) C")
    return gather_idx


def hilbert_permute(tokens, H, W, S, reverse=False):
    """Apply Hilbert permutation on patch tokens.

    Args:
        tokens: (..., H * W + S, C)
        N: number of frames
        H, W: height and width
        S: number of special tokens per frame
        reverse: if True, apply inverse Hilbert permutation

    Return:
        tokens: (..., H * W + S, C)
    """
    assert tokens.shape[-2] == H * W + S, f"{tokens.shape=}"
    C = tokens.shape[-1]

    # Squash batch dims into one dim
    batch_dims = tokens.shape[:-2]
    B_flat = math.prod(batch_dims) if len(batch_dims) > 0 else 1
    tokens = tokens.reshape(B_flat, H * W + S, C)

    # Separate patch and special tokens
    x_special = tokens[:, :S, :].contiguous()
    x_patch = tokens[:, S:, :].contiguous()  # (B * N, H * W, C)

    # Hilbert permutation
    gather_idx_o2h = get_hilbert_permutation(
        B_flat, 1, H, W, C, device=x_patch.device, inverse=reverse
    )
    x_patch = torch.gather(x_patch, dim=-2, index=gather_idx_o2h)

    # Combine patch and special tokens
    tokens = torch.cat([x_special, x_patch], dim=-2)
    tokens = tokens.reshape(batch_dims + (H * W + S, C))
    return tokens
