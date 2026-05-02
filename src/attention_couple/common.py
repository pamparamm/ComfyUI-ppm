import math
from typing import Any

import torch
import torch.nn.functional as F

COND = 0
UNCOND = 1

CondLike = list[tuple[torch.Tensor, dict[str, Any]]]


def lcm_for_list(numbers: list):
    current_lcm = numbers[0]
    for number in numbers[1:]:
        current_lcm = math.lcm(current_lcm, number)
    return current_lcm


def reshape_mask(mask: torch.Tensor, size: tuple[int, int], bs: int, num_tokens: int) -> torch.Tensor:
    num_conds = mask.shape[0]

    mask_downsample = F.interpolate(mask, size=size, mode="nearest")
    mask_downsample_reshaped = mask_downsample.view(num_conds, num_tokens, 1).repeat_interleave(bs, dim=0)

    return mask_downsample_reshaped
