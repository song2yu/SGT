from typing import List, Tuple

import torch
import torch.nn as nn

from einops import repeat
from diffusers.models.embeddings import get_1d_rotary_pos_embed


# def get_1d_rotary_pos_embed(dim: int, end: int, theta: int, freqs_dtype: torch.dtype) -> torch.Tensor:
#     # `freqs_dtype` should be torch.float32 or torch.float16 on NPU
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype)[: (dim // 2)] / dim))
#     t = torch.arange(end, dtype=freqs_dtype)
#     freqs = torch.einsum("i,j->ij", t, freqs)
#     # Reshape `freqs` to (end, dim) with real and imaginary parts interleaved
#     freqs_cos = freqs.cos()
#     freqs_sin = freqs.sin()
#     # The resulting freqs_cis is a float tensor with shape (end, dim)
#     freqs_cis = torch.stack([freqs_cos, freqs_sin], dim=-1).flatten(-2)
#     return freqs_cis

class OmniGen2RotaryPosEmbed(nn.Module):
    def __init__(self, theta: int,
                 axes_dim: Tuple[int, int, int],
                 axes_lens: Tuple[int, int, int] = (300, 512, 512),
                 patch_size: int = 2):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

    @staticmethod
    # def get_freqs_cis(axes_dim: Tuple[int, int, int],
    #                 axes_lens: Tuple[int, int, int],
    #                 theta: int) -> List[torch.Tensor]:
    #     freqs_cis = []
    #     # Note: use float32 uniformly on NPU
    #     freqs_dtype = torch.float32
        
    #     for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
    #         # 1. Call the diffusers helper, which returns a complex tensor
    #         #    For example, a tensor of shape (axis_len, axis_dim / 2) with dtype torch.complex64
    #         emb_complex = get_1d_rotary_pos_embed(d, e, theta=theta)
            
    #         # 2. Core change: convert the complex tensor into an NPU-friendly real-valued tensor
    #         #    `torch.view_as_real` turns (a+bi) into [a, b]
    #         #    Shape change: (L, D/2) -> (L, D/2, 2)
    #         emb_real = torch.view_as_real(emb_complex)
            
    #         #    Flatten the last two dims into an interleaved [cos, sin, cos, sin, ...] layout
    #         #    Shape change: (L, D/2, 2) -> (L, D)
    #         emb_real_interleaved = emb_real.flatten(start_dim=-2)
            
    #         freqs_cis.append(emb_real_interleaved)
            
    #     return freqs_cis
    def get_freqs_cis(axes_dim: Tuple[int, int, int],
                      axes_lens: Tuple[int, int, int],
                      theta: int) -> List[torch.Tensor]:
        freqs_cis = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
            emb = get_1d_rotary_pos_embed(d, e, theta=theta, freqs_dtype=freqs_dtype)
            freqs_cis.append(emb)
        return freqs_cis

    def _get_freqs_cis(self, freqs_cis, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if ids.device.type == "mps":
            ids = ids.to("cpu")

        result = []
        # import time
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()

        for i in range(len(self.axes_dim)):
            freqs = freqs_cis[i].to(ids.device)
            # Guard: Qwen/OmniGen2 RoPE tables are sized by ``axes_lens`` (10k
            # per axis by default). If any position id exceeds the table
            # length, the subsequent ``torch.gather`` triggers a CUDA
            # device-side assert that's hard to debug because it surfaces at
            # a later kernel. Do an upfront host-side check so we get an
            # actionable error message that identifies the offending axis,
            # max id, and table length.
            axis_ids = ids[:, :, i]
            max_id = int(axis_ids.max().item())
            if max_id >= freqs.shape[0]:
                raise RuntimeError(
                    f"[RoPE] position id out of range on axis {i}: "
                    f"max_id={max_id} >= axes_lens[{i}]={freqs.shape[0]}. "
                    f"Increase ``axes_lens`` in the model config or shrink "
                    f"the input sequence/image token count. "
                    f"(ids.shape={tuple(ids.shape)}, axis_ids.min="
                    f"{int(axis_ids.min().item())}, axis_ids.max={max_id})"
                )
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            # 1. Prepare inputs and convert to real representation
            input_tensor_complex = freqs.unsqueeze(0).repeat(index.shape[0], 1, 1)
            input_tensor_real = torch.view_as_real(input_tensor_complex)
            
            # 2. Expand index to gather both real and imaginary parts
            index_expanded = index.unsqueeze(-1).expand(-1, -1, -1, 2)
            
            # 3. Gather on the real-valued tensor
            gathered_real = torch.gather(input_tensor_real, dim=1, index=index_expanded)
            
            # 4. Change: do not convert back to complex here
            # gathered_complex = torch.view_as_complex(gathered_real) 
            
            # 5. Change: store the gathered real tensor directly in `result`
            result.append(gathered_real)

        concatenated_real = torch.cat(result, dim=-2)

        # 7. Change: convert back to complex only as the final step
        final_complex = torch.view_as_complex(concatenated_real)
        # torch.cuda.synchronize()
        # end_time = time.perf_counter()
        # elapsed_time = end_time - start_time
        # print(f"Complex-op elapsed time: {elapsed_time:.6f} sec")

            # result.append(torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index))
        return final_complex.to(device) # torch.Size([1, 5802, 60])
        # return torch.cat(result, dim=-1).to(device)

    def forward(
        self,
        freqs_cis,
        attention_mask,
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
        device
    ):
        batch_size = len(attention_mask)
        p = self.patch_size

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()

        seq_lengths = [cap_len + sum(ref_img_len) + img_len for cap_len, ref_img_len, img_len in zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len)]

        max_seq_len = max(seq_lengths)
        max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
        max_img_len = max(l_effective_img_len)

        # Create position IDs
        position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            # add text position ids
            position_ids[i, :cap_seq_len] = repeat(torch.arange(cap_seq_len, dtype=torch.int32, device=device), "l -> l 3")

            pe_shift = cap_seq_len
            pe_shift_len = cap_seq_len

            if ref_img_sizes[i] is not None:
                for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
                    H, W = ref_img_size
                    ref_H_tokens, ref_W_tokens = H // p, W // p
                    assert ref_H_tokens * ref_W_tokens == ref_img_len
                    # add image position ids

                    row_ids = repeat(torch.arange(ref_H_tokens, dtype=torch.int32, device=device), "h -> h w", w=ref_W_tokens).flatten()
                    col_ids = repeat(torch.arange(ref_W_tokens, dtype=torch.int32, device=device), "w -> h w", h=ref_H_tokens).flatten()
                    position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 0] = pe_shift
                    position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 1] = row_ids
                    position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 2] = col_ids

                    pe_shift += max(ref_H_tokens, ref_W_tokens)
                    pe_shift_len += ref_img_len

            H, W = img_sizes[i]
            H_tokens, W_tokens = H // p, W // p
            assert H_tokens * W_tokens == l_effective_img_len[i]

            row_ids = repeat(torch.arange(H_tokens, dtype=torch.int32, device=device), "h -> h w", w=W_tokens).flatten()
            col_ids = repeat(torch.arange(W_tokens, dtype=torch.int32, device=device), "w -> h w", h=H_tokens).flatten()

            assert pe_shift_len + l_effective_img_len[i] == seq_len
            position_ids[i, pe_shift_len: seq_len, 0] = pe_shift
            position_ids[i, pe_shift_len: seq_len, 1] = row_ids
            position_ids[i, pe_shift_len: seq_len, 2] = col_ids

        # Get combined rotary embeddings
        freqs_cis = self._get_freqs_cis(freqs_cis, position_ids)
        
        # create separate rotary embeddings for captions and images
        cap_freqs_cis = torch.zeros(
            batch_size, encoder_seq_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )
        ref_img_freqs_cis = torch.zeros(
            batch_size, max_ref_img_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )
        img_freqs_cis = torch.zeros(
            batch_size, max_img_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )

        for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len, seq_lengths)):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            ref_img_freqs_cis[i, :sum(ref_img_len)] = freqs_cis[i, cap_seq_len:cap_seq_len + sum(ref_img_len)]
            img_freqs_cis[i, :img_len] = freqs_cis[i, cap_seq_len + sum(ref_img_len):cap_seq_len + sum(ref_img_len) + img_len]

        return (
            cap_freqs_cis,
            ref_img_freqs_cis,
            img_freqs_cis,
            freqs_cis,
            l_effective_cap_len,
            seq_lengths,
        )