# Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.
#
# This modified file is released under the same license.


from dataclasses import dataclass
import warnings
from functools import partial
from typing import List, Optional, Tuple
import pdb
import sys
import os
import torch
from torch import nn
import numpy as np
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.functional import scaled_dot_product_attention
from transformers.utils import ModelOutput
from PIL import Image
import matplotlib.cm as cm  # Only the colormap is used, not pyplot.

from modeling.bagel.import_utils import is_flash_attn_available
if is_flash_attn_available():
    from flash_attn import flash_attn_varlen_func
else:
    warnings.warn("Cannot import flash_attn, install flash_attn to use Flash2Varlen attention for better performance")
import torch.nn.functional as F

from modeling.qwen2.modeling_qwen2 import (
    Qwen2Attention, 
    Qwen2MLP, 
    Qwen2PreTrainedModel, 
    Qwen2RMSNorm, 
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)

from modeling.qwen2.configuration_qwen2 import Qwen2Config as _Qwen2Config


IMAGE_GRID_SHAPE = (37, 37)
TOTAL_KV_TOKENS = 1371
TOTAL_IMAGE_TOKENS = IMAGE_GRID_SHAPE[0] * IMAGE_GRID_SHAPE[1] # 1369
DEFAULT_IMAGE_TOKEN_SLICE = slice(0, TOTAL_IMAGE_TOKENS)  # Default: [0:1369]

torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096
# flex_attention = torch.compile(flex_attention) # , dynamic=True, mode='max-autotune'
flex_attention = torch.compile(flex_attention)


class Qwen2Config(_Qwen2Config):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        is_causal=True,
        _attn_implementation="flash_attention_2",
        qk_norm=True,
        layer_module="Qwen2DecoderLayer",
        freeze_und=False,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            is_causal=is_causal,
            _attn_implementation=_attn_implementation,
            **kwargs,
        )
        self.qk_norm = qk_norm
        self.layer_module = layer_module
        self.freeze_und = freeze_und


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_lens(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None


def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


class PackedAttention(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask: List[torch.Tensor],
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ):
        # pdb.set_trace()
        packed_query_states = self.q_proj(packed_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        if isinstance(attention_mask, List):
            packed_key_states = packed_key_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states = packed_key_states.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0), 
                        key_states.to(torch.bfloat16).unsqueeze(0), 
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states = pad_sequence(packed_query_states.permute(1, 0, 2), pad_size)
            packed_key_states = pad_sequence(packed_key_states.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states.unsqueeze(0), 
                packed_key_states.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        return packed_attn_output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ):
        packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class PackedAttentionMoT(Qwen2Attention):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        if self.config.qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_moe_gen = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],# tuple: 33961 x 128 & 33961 x 128
        packed_und_token_indexes: torch.LongTensor, # 7337
        packed_gen_token_indexes: torch.LongTensor, # 26624
    ):
        
        packed_query_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_heads * self.head_dim))
        packed_key_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))
        packed_value_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_key_value_heads * self.head_dim))

        packed_sequence_und = packed_sequence[packed_und_token_indexes]
        packed_sequence_gen = packed_sequence[packed_gen_token_indexes]
        # pdb.set_trace()
        # Route the understanding / generation embeddings through their own experts.
        packed_query_states[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
        packed_query_states[packed_gen_token_indexes] = self.q_proj_moe_gen(packed_sequence_gen)

        packed_key_states[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
        packed_key_states[packed_gen_token_indexes] = self.k_proj_moe_gen(packed_sequence_gen)

        packed_value_states[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
        packed_value_states[packed_gen_token_indexes] = self.v_proj_moe_gen(packed_sequence_gen)

        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
        # self.config.freeze_und == True: all QKV layers have been detached, so these params do not receive updates.
        if self.config.freeze_und:
            packed_value_states[packed_und_token_indexes] = packed_value_states[packed_und_token_indexes].detach()

        packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
        packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

        packed_query_states_[packed_und_token_indexes] = self.q_norm(packed_query_states[packed_und_token_indexes])
        if self.config.freeze_und:
            packed_query_states_[packed_und_token_indexes] = packed_query_states_[packed_und_token_indexes].detach()
        packed_query_states_[packed_gen_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_gen_token_indexes])

        packed_key_states_[packed_und_token_indexes] = self.k_norm(packed_key_states[packed_und_token_indexes])
        if self.config.freeze_und:
            packed_key_states_[packed_und_token_indexes] = packed_key_states_[packed_und_token_indexes].detach()
        packed_key_states_[packed_gen_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_gen_token_indexes])

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
            packed_query_states_, packed_key_states_, packed_cos, packed_sin, unsqueeze_dim=1
        )

        if isinstance(attention_mask, List):
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            upacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0), 
                        key_states.to(torch.bfloat16).unsqueeze(0), 
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                        #  if "torch_npu" not in sys.modules else torch.logical_not(attention_mask_per_sample.bool()).unsqueeze(0)
                    )
                upacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(upacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states_ = pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
            packed_key_states_ = pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
            packed_value_states = pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = flex_attention(
                packed_query_states_.unsqueeze(0), # 1, num_head, L, head_dim
                packed_key_states_.unsqueeze(0), 
                packed_value_states.unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        packed_attn_output_[packed_und_token_indexes] = self.o_proj(packed_attn_output[packed_und_token_indexes])
        packed_attn_output_[packed_gen_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_gen_token_indexes])

        return packed_attn_output_

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ):
        if mode == 'und':
            packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
            packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
            packed_query_states = self.q_norm(packed_query_states)
            packed_key_states = self.k_norm(packed_key_states)
        elif mode == 'gen':
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            packed_query_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_heads * self.head_dim))
            packed_key_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))
            packed_value_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))

            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

            packed_query_states[packed_text_indexes] = self.q_proj(packed_text_query_sequence)
            packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(packed_vae_query_sequence)

            packed_key_states[packed_text_indexes] = self.k_proj(packed_text_query_sequence)
            packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(packed_vae_query_sequence)

            packed_value_states[packed_text_indexes] = self.v_proj(packed_text_query_sequence)
            packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(packed_vae_query_sequence)

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

            packed_query_states = packed_query_states.to(torch.float32)
            packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
            packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_vae_token_indexes])

            packed_key_states = packed_key_states.to(torch.float32)
            packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
            packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_vae_token_indexes])

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_value_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = flash_attn_varlen_func(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        if mode == 'und':
            packed_attn_output = self.o_proj(packed_attn_output)
        elif mode == 'gen':
            packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
            packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes])

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        packed_query_sequence = self.mlp(packed_query_sequence)
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


class Qwen2MoTDecoderLayer(nn.Module):
    token_counter = 1
    need_visual = False
    def __init__(
        self, 
        config, 
        layer_idx: Optional[int] = None, 
        attn_module: Optional[Qwen2Attention] = PackedAttentionMoT,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.freeze_und = config.freeze_und

        self.self_attn = attn_module(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def get_gqa_attention_map(self, 
        text_hidden_states: torch.Tensor,
        q_proj_layer: nn.Linear,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Manually compute the GQA cross-attention map.
        
        This mirrors how the VLM's text query attends to the image KV cache.

        Args:
            text_hidden_states (torch.Tensor): 
                Text hidden states (X), coming from packed_text_embedding.
                Shape: [N_q, D_model] -> [12, 3584]
                
            q_proj_layer (nn.Linear): 
                The Q projection layer (e.g. self.self_attn.q_proj).
                *Required* for turning X into Q.
                
            key_cache (torch.Tensor): 
                K cache from past_key_values.key_cache[0].
                Shape: [N_k, H_k, D_k] -> [1371, 4, 128]
                
            value_cache (torch.Tensor, optional): 
                V cache from past_key_values.value_cache[0].
                **Note: computing the attention *map* does not require V,
                but we list it to match the caller's inputs.**

        Returns:
            torch.Tensor: 
                The computed attention map.
                Shape: (B, H_q, N_q, N_k) -> [1, 28, 12, 1371]
        """
        NUM_Q_HEADS = 28
        NUM_KV_HEADS = 4
        HEAD_DIM = 128
        GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS  # 28 // 4 = 7


        # --- Step 1: compute the Q (query) vectors. ---
        # text_hidden_states (X) shape: [12, 3584]
        
        # Pass through the q_proj layer (Linear(3584, 3584)).
        q_projected = q_proj_layer(text_hidden_states)
        # q_projected shape: [12, 3584]

        # Reshape Q into the multi-head attention layout.
        # (N_q, D_model) -> (N_q, H_q, D_k)
        N_q = text_hidden_states.shape[0]  # 12
        q_states = q_projected.view(N_q, NUM_Q_HEADS, HEAD_DIM)
        # q_states shape: [12, 28, 128]
        
        print(f"Q states computed and reshaped: {q_states.shape}")

        # --- Step 2: prepare the K (key) vectors. ---
        # key_cache (K) shape: [1371, 4, 128]
        k_states = key_cache
        
        # --- Step 3: arrange Q and K for the batched matmul. ---
        # Target layout: (batch_size, num_heads, seq_len, head_dim).
        # We assume batch_size (B) = 1.
        
        # Q: (N_q, H_q, D_k) -> (B, H_q, N_q, D_k)
        # [12, 28, 128] -> [1, 28, 12, 128]
        q_calc = q_states.permute(1, 0, 2).unsqueeze(0)
        
        # K: (N_k, H_k, D_k) -> (B, H_k, N_k, D_k)
        # [1371, 4, 128] -> [1, 4, 1371, 128]
        k_calc = k_states.permute(1, 0, 2).unsqueeze(0)
        
        print(f"Q after layout change (B,H_q,N_q,D): {q_calc.shape}")
        print(f"K after layout change (B,H_k,N_k,D): {k_calc.shape}")

        # --- Step 4: handle GQA by repeating K heads. ---
        # (B, H_k, N_k, D_k) -> (B, H_q, N_k, D_k)
        # [1, 4, 1371, 128] -> [1, 28, 1371, 128]
        k_repeated = k_calc.repeat_interleave(GQA_RATIO, dim=1)
        
        print(f"K (GQA) after repeat (B,H_q,N_k,D): {k_repeated.shape}")

        # --- Step 5: compute attention scores (Q @ K.T). ---
        # Scale factor.
        scale = 1.0 / (HEAD_DIM ** 0.5)
        
        # Q (1, 28, 12, 128) @ K.T (1, 28, 128, 1371)
        scores = torch.matmul(q_calc, k_repeated.transpose(-2, -1))
        scaled_scores = scores * scale
        
        print(f"Attention scores shape (B,H_q,N_q,N_k): {scaled_scores.shape}")

        # --- Step 6: softmax -> final attention map. ---
        attention_map = F.softmax(scaled_scores, dim=-1)
        
        print("--- done ---")
        
        # Final shape: [1, 28, 12, 1371]
        return attention_map

    def save_attention_heatmap(self, 
        attention_map: torch.Tensor,
        original_image_pil: Image.Image,
        save_path: str,  # Where to save the heatmap file.
        head_index: int,
        token_index: int,
        grid_shape: tuple = IMAGE_GRID_SHAPE,
        image_token_slice: slice = DEFAULT_IMAGE_TOKEN_SLICE,
        alpha: float = 0.6
    ):
        """
        (Server-friendly version.)
        Render an attention map as a heatmap overlay on the original image and save it to disk.

        Args:
            attention_map (torch.Tensor): 
                Shape: [1, 28, 12, 1371] (B, H_q, N_q, N_k)
                
            original_image_pil (PIL.Image.Image): 
                The original 224x224 image.
                
            save_path (str): 
                Full output path for the heatmap, e.g. './heatmaps/head_0_token_5.png'
                
            head_index (int): index of the Q head to visualize (0 to 27).
            token_index (int): index of the text token to visualize (0 to 11).
            
            grid_shape (tuple): 2D grid shape of image patches, default (37, 37).
            image_token_slice (slice): slice that selects image tokens out of the 1371-dim axis.
            alpha (float): transparency of the heatmap overlay.
        """
        
        print(f"--- processing head={head_index}, token={token_index} ---")
        
        # --- Step 1: extract the 1D attention vector. ---
        try:
            attn_vector_all = attention_map[0, head_index, token_index, :].squeeze()
        except IndexError:
            print("Error: index out of range.")
            print(f"  Head range: 0-{attention_map.shape[1]-1}")
            print(f"  Token range: 0-{attention_map.shape[2]-1}")
            return

        # --- Step 2: keep only image tokens. ---
        attn_vector_image = attn_vector_all[image_token_slice]
        
        expected_tokens = grid_shape[0] * grid_shape[1]
        if attn_vector_image.shape[0] != expected_tokens:
            print(f"!! warning: sliced token count ({attn_vector_image.shape[0]})")
            print(f"   does not match grid_shape ({expected_tokens}).")
            print("   please check the 'image_token_slice' argument!")
            return

        # --- Step 3: reshape into a 2D heatmap. ---
        heatmap_2d = attn_vector_image.reshape(grid_shape)
        heatmap_2d_numpy = heatmap_2d.detach().cpu().numpy()

        # --- Step 4: resize the heatmap back to the original image size. ---
        orig_w, orig_h = original_image_pil.size # (224, 224)
        heatmap_tensor = torch.tensor(heatmap_2d_numpy).unsqueeze(0).unsqueeze(0)
        
        upscaled_heatmap_tensor = F.interpolate(
            heatmap_tensor,
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )
        upscaled_heatmap = upscaled_heatmap_tensor.squeeze().numpy() # (224, 224)

        # --- Step 5: normalize and colorize. ---
        norm_heatmap = (upscaled_heatmap - np.min(upscaled_heatmap)) / \
                    (np.max(upscaled_heatmap) - np.min(upscaled_heatmap) + 1e-6)
        
        colored_heatmap = cm.jet(norm_heatmap)[:, :, :3]  # Use the 'jet' colormap.
        colored_heatmap_pil = Image.fromarray((colored_heatmap * 255).astype(np.uint8))

        # --- Step 6: overlay the heatmap on the image. ---
        rgba_image = original_image_pil.convert('RGBA')
        rgba_heatmap = colored_heatmap_pil.convert('RGBA')
        rgba_heatmap.putalpha(int(255 * alpha))
        overlay_image = Image.alpha_composite(rgba_image, rgba_heatmap)

        # --- Step 7: save to disk (instead of showing). ---
        try:
            # Make sure the output directory exists.
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Convert the final RGBA image back to RGB and save.
            overlay_image.convert('RGB').save(save_path)
            print(f"+++ done! heatmap saved to: {save_path} +++")
        except Exception as e:
            print(f"!! error: failed to save file: {e} !!")
            
    def save_full_sentence_heatmap(self, 
        attention_map: torch.Tensor,
        original_image_pil: Image.Image,
        save_path: str,
        head_agg_method: str = "mean",     # either "mean" or "max"
        token_agg_method: str = "mean",   # either "mean" or "max"
        grid_shape: tuple = IMAGE_GRID_SHAPE,
        image_token_slice: slice = DEFAULT_IMAGE_TOKEN_SLICE,
        alpha: float = 0.6
    ):
        """
        (Server-friendly version -- aggregates everything.)
        Aggregate the attention map across *all heads* and *all tokens*,
        render it as a heatmap, and save it to disk.
        No head_index / token_index are needed.

        Args:
            attention_map (torch.Tensor): 
                Shape: [1, 28, 12, 1371] (B, H_q, N_q, N_k)
                
            original_image_pil (PIL.Image.Image): 
                The original 224x224 image.
                
            save_path (str): 
                Full save path for the heatmap, e.g. './heatmaps/full_sentence_mean.png'
                
            head_agg_method (str): how to aggregate the 28-head dim.
            token_agg_method (str): how to aggregate the 12-token dim.
            
            grid_shape, image_token_slice, alpha... (same as the single-token version)
        """
        
        print(f"--- aggregating heads ({head_agg_method}) and tokens ({token_agg_method}) ---")
        
        # --- Step 1: aggregate over heads and tokens. ---
        
        # Original shape: (B, H_q, N_q, N_k) -> (1, 28, 12, 1371)
        # Squeeze out the B dim.
        attn_map = attention_map.squeeze(0)  # Shape: [28, 12, 1371]

        # 1a. Aggregate over heads (dim=0).
        if head_agg_method == "mean":
            attn_map = torch.mean(attn_map, dim=0)
        elif head_agg_method == "max":
            attn_map, _ = torch.max(attn_map, dim=0)
        else:
            print(f"Error: unknown head_agg_method: {head_agg_method}")
            return
        # Shape after aggregation: [12, 1371] (N_q, N_k)

        # 1b. Aggregate over tokens (dim=0).
        if token_agg_method == "mean":
            attn_vector_all = torch.mean(attn_map, dim=0)
        elif token_agg_method == "max":
            attn_vector_all, _ = torch.max(attn_map, dim=0)
        else:
            print(f"Error: unknown token_agg_method: {token_agg_method}")
            return
        # Shape after aggregation: [1371] (N_k)

        # --- Steps 2-7 are identical to the single-head / single-token version. ---

        # --- Step 2: keep only image tokens. ---
        attn_vector_image = attn_vector_all[image_token_slice]
        expected_tokens = grid_shape[0] * grid_shape[1]
        if attn_vector_image.shape[0] != expected_tokens:
            print(f"!! warning: sliced token count ({attn_vector_image.shape[0]})")
            print(f"   does not match grid_shape ({expected_tokens}).")
            return

        # --- Step 3: reshape into a 2D heatmap. ---
        heatmap_2d = attn_vector_image.reshape(grid_shape)
        heatmap_2d_numpy = heatmap_2d.detach().cpu().numpy()

        # --- Step 4: resize the heatmap back to the original image size. ---
        orig_w, orig_h = original_image_pil.size # (224, 224)
        heatmap_tensor = torch.tensor(heatmap_2d_numpy).unsqueeze(0).unsqueeze(0)
        
        upscaled_heatmap_tensor = F.interpolate(
            heatmap_tensor,
            size=(orig_h, orig_h),
            mode='bilinear',
            align_corners=False
        )
        upscaled_heatmap = upscaled_heatmap_tensor.squeeze().numpy()

        # --- Step 5: normalize and colorize. ---
        norm_heatmap = (upscaled_heatmap - np.min(upscaled_heatmap)) / (np.max(upscaled_heatmap) - np.min(upscaled_heatmap) + 1e-6)
        colored_heatmap = cm.jet(norm_heatmap)[:, :, :3]
        colored_heatmap_pil = Image.fromarray((colored_heatmap * 255).astype(np.uint8))

        # --- Step 6: overlay the heatmap on the image. ---
        rgba_image = original_image_pil.convert('RGBA')
        rgba_heatmap = colored_heatmap_pil.convert('RGBA')
        rgba_heatmap.putalpha(int(255 * alpha))
        overlay_image = Image.alpha_composite(rgba_image, rgba_heatmap)

        # --- Step 7: save to disk. ---
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            overlay_image.convert('RGB').save(save_path)
            print(f"+++ done! aggregated heatmap saved to: {save_path} +++")
        except Exception as e:
            print(f"!! error: failed to save file: {e} !!")


    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.input_layernorm(packed_sequence[packed_und_token_indexes])
        packed_sequence_[packed_gen_token_indexes] = self.input_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])

        # Self Attention
        packed_sequence_ = self.self_attn(
            packed_sequence=packed_sequence_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
        packed_sequence = residual + packed_sequence_

        # Fully Connected
        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_[packed_und_token_indexes] = self.mlp(
            self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
    
        packed_sequence_[packed_gen_token_indexes] = self.mlp_moe_gen(
            self.post_attention_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])
        )
        packed_sequence = residual + packed_sequence_

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.input_layernorm(packed_query_sequence)
        elif mode == "gen":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            packed_query_sequence_[packed_text_indexes] = self.input_layernorm(packed_query_sequence[packed_text_indexes])
            packed_query_sequence_[packed_vae_token_indexes] = self.input_layernorm_moe_gen(packed_query_sequence[packed_vae_token_indexes])
            packed_query_sequence = packed_query_sequence_

        # Self Attention
        # packed_query_sequence:12x3584   past_key_values.key_cache[0]   past_key_values.value_cache[0]

        hidden_states_for_viz = packed_query_sequence
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        if past_key_values.value_cache[2] is not None and Qwen2MoTDecoderLayer.need_visual:
            attn_map = self.get_gqa_attention_map(hidden_states_for_viz, self.self_attn.q_proj, past_key_values.key_cache[2][1:1370], past_key_values.value_cache[2][1:1370])
            original_image_pil = Image.open('analysis/mmvp/9.png')
            # save_path = 'analysis/MMVP_atten/20_8_4_27.png'
            save_path_agg = 'analysis/MMVP_atten/9_agg_Z_2_' + str(Qwen2MoTDecoderLayer.token_counter) + '.png'
            # self.save_attention_heatmap(attn_map, original_image_pil, save_path, head_index=8, token_index=4)
            self.save_full_sentence_heatmap(attn_map, original_image_pil, save_path_agg)
            Qwen2MoTDecoderLayer.token_counter += 1
            Qwen2MoTDecoderLayer.need_visual = False

        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
            packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(packed_vae_query_sequence).to(torch.bfloat16)

            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_text_query_sequence)
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_vae_query_sequence)
            packed_query_sequence = packed_query_sequence_

        packed_query_sequence = residual + packed_query_sequence
        return packed_query_sequence, past_key_values


class Qwen2MoEDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.mlp_moe_gen = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:

        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        # Self Attention
        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        # Fully Connected
        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)

        packed_sequence_new = packed_sequence.new_zeros(packed_sequence.shape)
        packed_sequence_und = self.mlp(packed_sequence[packed_und_token_indexes])
        packed_sequence_gen = self.mlp_moe_gen(packed_sequence[packed_gen_token_indexes])
        packed_sequence_new[packed_und_token_indexes] = packed_sequence_und
        packed_sequence_new[packed_gen_token_indexes] = packed_sequence_gen

        packed_sequence = residual + packed_sequence_new

        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        # Self Attention
        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        # Fully Connected
        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        if mode == "und":
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            packed_query_sequence_[packed_text_indexes] = self.mlp(packed_query_sequence[packed_text_indexes])
            packed_query_sequence_[packed_vae_token_indexes] = self.mlp_moe_gen(packed_query_sequence[packed_vae_token_indexes])
            packed_query_sequence = packed_query_sequence_
        packed_query_sequence = residual + packed_query_sequence

        return packed_query_sequence, past_key_values


Decoder_layer_dict = {
    "Qwen2DecoderLayer": Qwen2DecoderLayer,
    "Qwen2MoEDecoderLayer": Qwen2MoEDecoderLayer,
    "Qwen2MoTDecoderLayer": partial(Qwen2MoTDecoderLayer, attn_module=PackedAttentionMoT),
}


class Qwen2Model(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.use_moe = 'Mo' in config.layer_module

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layer_module = Decoder_layer_dict[config.layer_module]
        self.layers = nn.ModuleList(
            [layer_module(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        if self.config.freeze_und:
            packed_sequence[packed_und_token_indexes] = packed_sequence[packed_und_token_indexes].detach()

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            assert packed_und_token_indexes is not None
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )
        
        for decoder_layer in self.layers:
            packed_sequence = decoder_layer(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs
            )

        if self.use_moe:
            packed_sequence_ = torch.zeros_like(packed_sequence)
            packed_sequence_[packed_und_token_indexes] = self.norm(packed_sequence[packed_und_token_indexes])
            if self.config.freeze_und:
                packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
            packed_sequence_[packed_gen_token_indexes] = self.norm_moe_gen(packed_sequence[packed_gen_token_indexes])
            return packed_sequence_
        else:
            return self.norm(packed_sequence)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        # create position embeddings to be shared across the decoder layers
        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == 'gen':
                assert packed_vae_token_indexes is not None
                assert packed_text_indexes is not None
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )

        for decoder_layer in self.layers:
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )

        if self.use_moe:
            if mode == "und":
                packed_query_sequence = self.norm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                packed_query_sequence_[packed_text_indexes] = self.norm(packed_query_sequence[packed_text_indexes])
                packed_query_sequence_[packed_vae_token_indexes] = self.norm_moe_gen(packed_query_sequence[packed_vae_token_indexes])
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = self.norm(packed_query_sequence)

        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_moe(self):
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "")
                param.data.copy_(self.state_dict()[original_name].data)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:

        outputs = self.model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        return outputs

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values=True,
        is_causal=True,
        mode="und",
        packed_vae_token_indexes=None,
        packed_text_indexes=None,
    ) -> BaseNavitOutputWithPast:

        outputs = self.model(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )

        return outputs





def get_manual_attention_map(query_states, key_states):
    """
    Manually compute the GQA cross-attention map.
    
    Assumes the input shape is [total_tokens, num_heads, head_dim] with batch size = 1.
    
    Args:
        query_states (torch.Tensor): the Q tensor.
                                    Example shape: torch.Size([1, 28, 128])
        key_states (torch.Tensor): the K tensor.
                                   Example shape: torch.Size([1583, 4, 128])
                                   
    Returns:
        torch.Tensor: the computed attention map.
                      Shape: (B, H_q, N_q, N_k) -> (1, 28, 1, 1583)
    """
    
    print(f"Original Q shape (N_q, H_q, D): {query_states.shape}")
    print(f"Original K shape (N_k, H_k, D): {key_states.shape}")

    # --- 1. Determine the GQA repeat factor. ---
    num_q_heads = query_states.shape[1]  # 28
    num_k_heads = key_states.shape[1]    # 4
    
    # Make sure num_q_heads is divisible by num_k_heads.
    if num_q_heads % num_k_heads != 0:
        raise ValueError("Number of Q heads must be a multiple of the number of K/V heads.")
        
    gqa_ratio = num_q_heads // num_k_heads # 28 // 4 = 7
    print(f"GQA ratio: {gqa_ratio}")

    # --- 2. Adjust the data layout (permute & unsqueeze). ---
    # Target: convert [N, H, D] -> [B, H, N, D] with B=1.
    
    # Q: [1, 28, 128] -> [28, 1, 128] (permute) -> [1, 28, 1, 128] (unsqueeze)
    # B=1, H_q=28, N_q=1, D=128
    q_calc = query_states.permute(1, 0, 2).unsqueeze(0)
    
    # K: [1583, 4, 128] -> [4, 1583, 128] (permute) -> [1, 4, 1583, 128] (unsqueeze)
    # B=1, H_k=4, N_k=1583, D=128
    k_calc = key_states.permute(1, 0, 2).unsqueeze(0)
    
    print(f"Q shape for compute (B, H_q, N_q, D): {q_calc.shape}")
    print(f"K shape for compute (B, H_k, N_k, D): {k_calc.shape}")

    # --- 3. Handle GQA by repeating K heads. ---
    # Repeat K along dim=1 (H_k) ``gqa_ratio`` times.
    # [1, 4, 1583, 128] -> [1, 28, 1583, 128]
    k_repeated = k_calc.repeat_interleave(gqa_ratio, dim=1)
    print(f"K shape after repeat (B, H_q, N_k, D): {k_repeated.shape}")

    # --- 4. Compute the scale factor. ---
    head_dim = q_calc.shape[-1] # 128
    scale = 1.0 / (head_dim ** 0.5)

    # --- 5. Compute QK^T (attention scores). ---
    # Q (B, H, N_q, D) @ K.T (B, H, D, N_k)
    # (1, 28, 1, 128) @ (1, 28, 128, 1583)
    scores = torch.matmul(q_calc, k_repeated.transpose(-2, -1))
    
    # Output shape: (B, H, N_q, N_k) -> (1, 28, 1, 1583)
    print(f"Scores shape (B, H, N_q, N_k): {scores.shape}")

    # --- 6. Scale + softmax -> attention map. ---
    scaled_scores = scores * scale
    attention_map = F.softmax(scaled_scores, dim=-1)
    
    return attention_map


# my_attention_map = get_manual_attention_map(packed_query_states, merged_key_states)
