# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Tuple, Union

import torch
from torch import nn


from diffusers.models.activations import get_activation


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

        self.initialize_weights()
        
    def initialize_weights(self):
        nn.init.normal_(self.linear_1.weight, std=0.02)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.normal_(self.linear_2.weight, std=0.02)
        nn.init.zeros_(self.linear_2.bias)
        
    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample

def fast_complex_mul(a_c64: torch.Tensor, b_c64: torch.Tensor) -> torch.Tensor:
    """
    Performs a high-performance complex multiplication of two complex64 tensors
    by decomposing the operation into real-valued, NPU-friendly ops.

    Args:
        a_c64 (torch.Tensor): A tensor with dtype torch.complex64.
        b_c64 (torch.Tensor): A tensor with dtype torch.complex64.

    Returns:
        torch.Tensor: A real-valued tensor (dtype=torch.float32) representing
                      the complex result, with real/imag parts in the last dimension.
                      Shape: [..., D, 2].
    """
    # 1. Convert both complex64 tensors into their float32 real-valued representation
    #    This is an efficient memory-view operation with almost zero overhead
    #    Shape change: [..., D] -> [..., 2D]
    a_real_view = torch.view_as_real(a_c64).to(torch.float16)
    b_real_view = torch.view_as_real(b_c64).to(torch.float16)

    # 2. Split into real and imaginary parts
    a_re, a_im = a_real_view[..., 0], a_real_view[..., 1]
    b_re, b_im = b_real_view[..., 0], b_real_view[..., 1]

    # 3. Perform complex multiplication in the real domain using operations that NPUs optimize well
    #    (a_re + i*a_im) * (b_re + i*b_im) = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
    out_re = a_re * b_re - a_im * b_im
    out_im = a_re * b_im + a_im * b_re

    # 4. Merge the results back into a single real tensor with [real, imag] on the last dim
    out_real = torch.stack([out_re, out_im], dim=-1)
    
    return out_real.flatten(start_dim=-2)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen and CogView4
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        # x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # import time
        # torch.cuda.synchronize()
        # time_124 = time.perf_counter()

        x_rotated = torch.view_as_complex(x.to(torch.float32).reshape(*x.shape[:-1], x.shape[-1] // 2, 2)) # to(torch.float32)
        freqs_cis = freqs_cis.unsqueeze(2).to(torch.complex64)
        # time_129 = time.perf_counter()
        # print(f'-------------124-129: {time_129-time_124}')

        # x_out = fast_complex_mul(x_rotated, freqs_cis)
        x_out_64 = (x_rotated * freqs_cis).to(torch.complex64)
        x_out = torch.view_as_real(x_out_64).flatten(3)
        # time_131 = time.perf_counter()
        # print(f'-------------129-131: {time_131-time_129}')

        return x_out.type_as(x) # float32

