# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot on headless servers.
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union
import copy
from typing import List, Tuple, Optional
from typing import Optional, List, Union, Dict, Any, Tuple
import matplotlib.pyplot as plt
import pdb
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
import os
from PIL import Image
import matplotlib.cm as cm  # Only the colormap is used, not pyplot.

IMAGE_GRID_SHAPE = (37, 37)
TOTAL_KV_TOKENS = 1371
TOTAL_IMAGE_TOKENS = IMAGE_GRID_SHAPE[0] * IMAGE_GRID_SHAPE[1]  # 1369
DEFAULT_IMAGE_TOKEN_SLICE = slice(0, TOTAL_IMAGE_TOKENS)  # Default: [0:1369]

from data.data_utils import (
    create_sparse_mask, 
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    patchify, 
)
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding


class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift


class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # for visual understanding
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        # for visual generation
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            sequence_length: length of sequence.
            packed_text_ids: 1-D int tensor, packed text token ids.
            packed_text_indexes: 1-D int tensor, packed text token indexes in sequence.
            sample_lens: A list of N ints, length of each sample in packed_sequence.
            nested_attention_masks: A list of N 2-D float tensor,  where 0.0 means attention and 
                -inf means ignore.
            packed_position_ids: packed 1-D positions, an image has only one global position shared
                by all latent tokens.

            packed_vit_tokens: packed patchified image tokens for vit model.
            packed_vit_position_ids: 1-D int tensor, the position of each token for vit model. 
            packed_vit_token_indexes: 1-D int tensor, packed ViT token indexes in sequence.
            vit_token_seqlens: 1-D int tensor, the length of each image tokens for vit model.
            packed_label_ids: 1-D int tensor, packed label token ids.
            ce_loss_indexes: 1-D bool tensor, where to compute ce loss.

            padded_latent: padded latent from VAE encoder.
            patchified_vae_latent_shapes: A list of (h, w) tuples, patchfied latent shapes of each image.
            packed_latent_position_ids: 1-D int tensor, the position of each token for latent.
            packed_vae_token_indexes: 1-D int tensor, padded image token indexes in sequence.
            packed_timesteps: 1-D float tensor, flow timesteps. 0 indicates use clean image.
            mse_loss_indexes: 1-D bool tensor, where to compute mse loss.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model( # packed_vit_tokens: 6656 x 588
                packed_pixel_values=packed_vit_tokens, 
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)  # packed_vit_token_embed: 6656 x 1152 -->  6656 x 3584
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb # understanding tokens
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed # packed_vit_token_embed: 6656 x 3584
        
        if self.config.visual_gen and padded_latent is not None:
            p = self.latent_patch_size # 2
            packed_latent = []
            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes): # padded_latent: 26 x 16 x 64 x 64
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p) # latent: 16 x 64 x 64
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            # add noise and timestep to latent
            noise = torch.randn_like(packed_latent_clean)
            packed_timesteps = torch.sigmoid(packed_timesteps) 
            packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps) # shifted noise
            packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise # add noise
            packed_timestep_embeds = self.time_embedder(packed_timesteps)
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + latent_token_pos_emb
            packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )
        last_hidden_state = self.language_model( # last_hidden_state: 33961 x 3584
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        mse = None
        if self.config.visual_gen and padded_latent is not None:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes]) # self.llm2vae:3584-->64 33961(26624) x 3584 --> packed_mse_preds: 26624 x 64
            target = noise - packed_latent_clean # NOTE: v_t=dx_t/dt=x_1-x_0, pointing from data to noise
            has_mse = packed_timesteps > 0
            mse = (packed_mse_preds - target[has_mse]) ** 2

        ce = None
        # pdb.set_trace()
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")

        return dict(mse=mse, ce=ce)


    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def analyze_latent_to_text_attention(
        self,
        attn_map: torch.Tensor,           # [num_heads, query_len, kv_len]
        latent_start_in_query: int,       # start index of the latent tokens in the query
        latent_end_in_query: int,         # end index (exclusive) of the latent tokens in the query
        text_start_in_kv: int,            # start index of the text tokens in the KV cache
        text_end_in_kv: int,              # end index (exclusive) of the text tokens in the KV cache
        timestep: float = None,           # current diffusion timestep
        layer_idx: int = None,            # layer index
    ):
        """
        Analyse the attention that latent tokens (denoised image) pay to text tokens.
        
        Args:
            attn_map: attention map, shape [num_heads, query_seq_len, kv_seq_len]
            latent_start_in_query: start index of the latent tokens in the query sequence.
            latent_end_in_query: end index (exclusive) of the latent tokens in the query sequence.
            text_start_in_kv: start index of the text tokens in the KV cache.
            text_end_in_kv: end index (exclusive) of the text tokens in the KV cache.
            timestep: current denoising timestep.
            layer_idx: current layer index.
        
        Returns:
            dict: a dictionary of summary statistics.
        """
        # Extract the latent-to-text region: latent query attends to text key.
        # attn_map shape: [num_heads, query_len, kv_len]
        l2t_attn = attn_map[:, latent_start_in_query:latent_end_in_query, text_start_in_kv:text_end_in_kv]
        # l2t_attn shape: [num_heads, num_latent_tokens, num_text_tokens]
        
        # Average across all heads.
        l2t_attn_mean = l2t_attn.mean(dim=0)  # [num_latent_tokens, num_text_tokens]
        
        # Stats.
        l2t_values = l2t_attn_mean.flatten().detach().cpu().numpy()
        
        # Total attention each latent token pays to all text tokens.
        l2t_per_latent = l2t_attn_mean.sum(dim=-1)  # [num_latent_tokens]
        
        # Total attention each text token receives from all latent tokens.
        l2t_per_text = l2t_attn_mean.sum(dim=0)  # [num_text_tokens]
        
        # Compute the attention entropy (a spread measure).
        eps = 1e-10
        l2t_normalized = l2t_attn_mean / (l2t_attn_mean.sum(dim=-1, keepdim=True) + eps)
        l2t_normalized = l2t_normalized.clamp(min=eps)
        l2t_entropy = -torch.sum(l2t_normalized * torch.log(l2t_normalized), dim=-1).mean()
        
        # Theoretical maximum entropy (uniform distribution).
        num_text_tokens = text_end_in_kv - text_start_in_kv
        max_entropy = np.log(num_text_tokens)
        
        results = {
            'l2t_mean': l2t_values.mean(),
            'l2t_std': l2t_values.std(),
            'l2t_max': l2t_values.max(),
            'l2t_min': l2t_values.min(),
            'l2t_sum': l2t_values.sum(),
            'l2t_per_latent_mean': l2t_per_latent.mean().item(),
            'l2t_per_latent_std': l2t_per_latent.std().item(),
            'l2t_per_text_mean': l2t_per_text.mean().item(),
            'l2t_per_text_max': l2t_per_text.max().item(),
            'l2t_entropy': l2t_entropy.item(),
            'l2t_entropy_ratio': l2t_entropy.item() / max_entropy,
            'l2t_attn_matrix': l2t_attn_mean.detach().cpu(),  # Full matrix, kept for visualization.
        }
        
        # Log summary.
        prefix = f"[t={timestep:.3f}]" if timestep is not None else ""
        prefix += f"[layer={layer_idx}]" if layer_idx is not None else ""
        print(f"{prefix} L2T Attention: mean={results['l2t_mean']:.6f}, "
            f"per_latent_mean={results['l2t_per_latent_mean']:.4f}, "
            f"entropy_ratio={results['l2t_entropy_ratio']:.4f}")
        
        return results


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
        
        # print(f"Q states computed and reshaped: {q_states.shape}")

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
        
        # print(f"Q after layout change (B,H_q,N_q,D): {q_calc.shape}")
        # print(f"K after layout change (B,H_k,N_k,D): {k_calc.shape}")

        # --- Step 4: handle GQA by repeating K heads. ---
        # (B, H_k, N_k, D_k) -> (B, H_q, N_k, D_k)
        # [1, 4, 1371, 128] -> [1, 28, 1371, 128]
        k_repeated = k_calc.repeat_interleave(GQA_RATIO, dim=1)
        
        # print(f"K (GQA) after repeat (B,H_q,N_k,D): {k_repeated.shape}")

        # --- Step 5: compute attention scores (Q @ K.T). ---
        # Scale factor.
        scale = 1.0 / (HEAD_DIM ** 0.5)
        
        # Q (1, 28, 12, 128) @ K.T (1, 28, 128, 1371)
        scores = torch.matmul(q_calc, k_repeated.transpose(-2, -1))
        scaled_scores = scores * scale
        
        # print(f"Attention scores shape (B,H_q,N_q,N_k): {scaled_scores.shape}")

        # --- Step 6: softmax -> final attention map. ---
        attention_map = F.softmax(scaled_scores, dim=-1)
        
        # print('--- done ---')
        
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
            print(f"Error: index out of range.")
            print(f"  Head range: 0-{attention_map.shape[1]-1}")
            print(f"  Token range: 0-{attention_map.shape[2]-1}")
            return

        # --- Step 2: keep only image tokens. ---
        attn_vector_image = attn_vector_all[image_token_slice]
        
        expected_tokens = grid_shape[0] * grid_shape[1]
        if attn_vector_image.shape[0] != expected_tokens:
            print(f"!! warning: sliced token count ({attn_vector_image.shape[0]})")
            print(f"   does not match grid_shape ({expected_tokens}).")
            print(f"   please check the 'image_token_slice' argument!")
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
        head_agg_method: str = "max",     # either "mean" or "max"
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

    @torch.no_grad
    def compute_attention_map(
        self, 
        hidden_states,    # [seq_len, hidden_size] -- already has RoPE applied
        k_cache,          # [seq_len, num_kv_heads, head_dim] -- already has RoPE applied
        q_proj_weight,    # Q projection weight
        num_heads,        # number of Q heads (e.g. 28)
        num_kv_heads,     # number of KV heads (e.g. 4)
    ):
        seq_len, hidden_size = hidden_states.shape
        head_dim = k_cache.shape[-1]
        
        # 1. Q projection.
        q = F.linear(hidden_states, q_proj_weight)  # [seq_len, num_heads * head_dim]
        q = q.view(seq_len, num_heads, head_dim)    # [seq_len, num_heads, head_dim]
        q = q.permute(1, 0, 2)                       # [num_heads, seq_len, head_dim]
        
        # 2. Expand K for GQA.
        k = k_cache.permute(1, 0, 2)                             # [num_kv_heads, seq_len, head_dim]
        k = k.repeat_interleave(num_heads // num_kv_heads, dim=0) # [num_heads, seq_len, head_dim]
        
        # 3. Attention = softmax(Q @ K^T / sqrt(d))
        attn = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        
        # 4. Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=attn.device), diagonal=1).bool()
        # attn = attn.masked_fill(mask, float('-inf'))
        
        # 5. Softmax
        attn = F.softmax(attn, dim=-1)
        
        return attn  # [num_heads, seq_len, seq_len]

    def analyze_svd(self, tokens, name="tokens", top_k=50, layer_index=0):
        """
        Run SVD analysis on visual tokens.
        
        Args:
            tokens: shape (num_tokens, hidden_dim) or (batch, num_tokens, hidden_dim)
            name: a display-only name.
            top_k: show the top-k singular values.
        """
        # Convert dtype.
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().float().numpy()
        if tokens.ndim == 3:
            tokens = tokens.reshape(-1, tokens.shape[-1])
        
        # SVD
        U, S, Vh = np.linalg.svd(tokens, full_matrices=False)
        
        # Compute metrics.
        explained_var = (S ** 2) / np.sum(S ** 2)
        cumsum_var = np.cumsum(explained_var)
        
        # Effective rank.
        norm_sv = S / np.sum(S)
        effective_rank = np.exp(-np.sum(norm_sv * np.log(norm_sv + 1e-10)))
        
        # Number of components needed to reach 90 / 95 / 99% variance.
        var_90 = np.searchsorted(cumsum_var, 0.90) + 1
        var_95 = np.searchsorted(cumsum_var, 0.95) + 1
        var_99 = np.searchsorted(cumsum_var, 0.99) + 1
        
        # Print results.
        print(f"layer index:[{layer_index}] Shape: {tokens.shape} | Effective Rank: {effective_rank:.1f} | "
            f"90%: {var_90}, 95%: {var_95}, 99%: {var_99}")
        
        return {'S': S, 'cumsum_var': cumsum_var, 'effective_rank': effective_rank}


    def analyze_attn_regions(self, attn_sum, vision_end=1369):
        """
        Analyse attention over different regions.
        Sequence layout: [vision tokens (0 to vision_end-1)] [text tokens (vision_end to end)].
        """
        attn_np = attn_sum.detach().cpu().numpy()
        seq_len = attn_np.shape[0]
        text_len = seq_len - vision_end
        
        # === Extract each region. ===
        
        # V2V: vision query -> vision key (all valid).
        v2v = attn_np[:vision_end, :vision_end]
        v2v_values = v2v.flatten() 
        
        # T2V: text query -> vision key (all valid, no causal mask).
        # t2v = attn_np[vision_end:, :vision_end]
        t2v = attn_np[-4:-2, :vision_end]
        # t2v = t2v[np.newaxis, :]
        t2v_values = t2v.flatten()
        
        # T2T: text query -> text key (lower-triangular valid).
        t2t = attn_np[vision_end:, vision_end:]
        t2t_mask = np.tril(np.ones_like(t2t, dtype=bool))
        t2t_values = t2t[t2t_mask]
        
        # === Log stats. ===
        # print("=" * 50)
        # print(f"Vision: 0 ~ {vision_end-1} ({vision_end} tokens)")
        # print(f"Text:   {vision_end} ~ {seq_len-1} ({text_len} tokens)")
        # print("=" * 50)
        
        # print(f"\n[V → V] Vision attend to Vision:")
        # print(f"  mean: {v2v_values.mean():.6f}")
        # print(f"  sum:  {v2v_values.sum():.2f}")
        
        # print(f"\n[T → V] Text attend to Vision:")
        # print(f"  mean: {t2v_values.mean():.6f}")
        # print(f"  sum:  {t2v_values.sum():.2f}")
        
        # print(f"\n[T → T] Text attend to Text:")
        # print(f"  mean: {t2t_values.mean():.6f}")
        # print(f"  sum:  {t2t_values.sum():.2f}")
        
        # === Average amount of attention each text token pays to vision. ===
        t2v_per_token = t2v.sum(axis=1)  # total attention each text token pays to all vision tokens
        # print(f"\n[T -> V] total attention per text token to vision:")
        # print(f"  mean: {t2v_per_token.mean():.4f}")
        # print(f"  max:  {t2v_per_token.max():.4f}")
        # print(f"  min:  {t2v_per_token.min():.4f}")
        
        return {
            'v2v': v2v_values,
            't2v': t2v_values,
            't2t': t2t_values,
            't2v_matrix': t2v,  # (text_len, vision_end) matrix
            'avg_t2v':t2v_per_token.mean()
        }

    def compute_attention_entropy(self, attn_map, vision_end, mode, eps=1e-10):
        """
        Compute and log the attention entropy.
        
        Args:
            attn_map: attention map, shape [seq_len, seq_len] or [num_heads, seq_len, seq_len]
            vision_end: end index of vision tokens.
            eps: small constant to prevent log(0).
        """
        if attn_map.dim() == 2:
            attn_map = attn_map.unsqueeze(0)
        
        seq_len = attn_map.shape[-1]
        
        def calc_entropy(attn):
            attn = attn / (attn.sum(dim=-1, keepdim=True) + eps)
            attn = attn.clamp(min=eps)
            return -torch.sum(attn * torch.log(attn), dim=-1).mean()
        
        # Vision to Vision
        v2v_attn = attn_map[:, :vision_end, :vision_end]
        v2v_entropy = calc_entropy(v2v_attn).item()
        
        # Text to Vision
        t2v_attn = attn_map[:, vision_end:, :vision_end]
        t2v_entropy = calc_entropy(t2v_attn).item()
        
        # Theoretical maximum entropy.
        max_entropy = torch.log(torch.tensor(vision_end, dtype=torch.float)).item()
        # if mode=='v2v':
        #     print(f"V2V Entropy: {v2v_entropy:.4f} (max: {max_entropy:.4f}, ratio: {v2v_entropy/max_entropy:.4f})")
        # else:
        #     print(f"T2V Entropy: {t2v_entropy:.4f} (max: {max_entropy:.4f}, ratio: {t2v_entropy/max_entropy:.4f})")
        
        return v2v_entropy, t2v_entropy

    def visualize_token_similarity(
        self,
        hidden_states: Union[torch.Tensor, np.ndarray],
        save_path: str,
        num_text_tokens: Optional[int] = None,
        num_visual_tokens: int = 1371,
        text_first: bool = True,
        token_labels: Optional[List[str]] = None,
        figsize: tuple = (12, 10),
        cmap: str = "RdBu_r",
        title: str = "Token Cosine Similarity Matrix",
        show_values: bool = False,
        show_boundary: bool = True,
        dpi: int = 150,
    ):
        """
        Compute and visualize cosine similarity between tokens, then save the plot.
        
        Args:
            hidden_states: tensor of shape (num_tokens, hidden_dim) or (batch, num_tokens, hidden_dim).
            save_path: path to save the figure.
            num_text_tokens: number of text tokens (optional, used for a divider line).
            num_visual_tokens: number of visual tokens (optional).
            text_first: whether text tokens come first.
            token_labels: label for each token.
            figsize: figure size.
            cmap: colormap.
            title: figure title.
            show_values: whether to render numeric values (recommended only when #tokens <= 30).
            show_boundary: whether to draw the text/visual divider.
            dpi: figure dpi.
        
        Returns:
            similarity_matrix: cosine similarity matrix (numpy array).
        """
        
        # Convert to torch tensor.
        if isinstance(hidden_states, np.ndarray):
            hidden_states = torch.from_numpy(hidden_states)
        
        # Handle dimensions.
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[0]  # take the first batch
        
        hidden_states = hidden_states.float()
        
        # Normalize, then compute cosine similarity.
        hidden_states_norm = F.normalize(hidden_states, p=2, dim=-1)
        similarity_matrix = torch.mm(hidden_states_norm, hidden_states_norm.t()).cpu().numpy()
        
        num_tokens = similarity_matrix.shape[0]
        
        # Create the figure.
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw the heatmap.
        sns.heatmap(
            similarity_matrix,
            ax=ax,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            annot=show_values and num_tokens <= 30,
            fmt=".2f",
            square=True,
            cbar_kws={"label": "Cosine Similarity"},
            xticklabels=token_labels if token_labels else False,
            yticklabels=token_labels if token_labels else False,
        )
        
        # Draw the divider.
        if show_boundary and num_text_tokens is not None:
            boundary = num_text_tokens if text_first else num_visual_tokens
            ax.axhline(y=boundary, color='lime', linewidth=2, linestyle='--')
            ax.axvline(x=boundary, color='lime', linewidth=2, linestyle='--')
        
        ax.set_xlabel("Token Index", fontsize=12)
        ax.set_ylabel("Token Index", fontsize=12)
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        # Log stats.
        print(f"Figure saved to: {save_path}")
        print(f"Similarity stats: mean={similarity_matrix.mean():.4f}, std={similarity_matrix.std():.4f}")
        
        # If text/visual counts were provided, also log region-wise stats.
        if num_text_tokens is not None and num_visual_tokens is not None:
            if text_first:
                t_start, t_end = 0, num_text_tokens
                v_start, v_end = num_text_tokens, num_text_tokens + num_visual_tokens
            else:
                v_start, v_end = 0, num_visual_tokens
                t_start, t_end = num_visual_tokens, num_visual_tokens + num_text_tokens
            
            tt = similarity_matrix[t_start:t_end, t_start:t_end]
            vv = similarity_matrix[v_start:v_end, v_start:v_end]
            tv = similarity_matrix[t_start:t_end, v_start:v_end]
            
            print(f"Text-Text:     mean={tt.mean():.4f}, std={tt.std():.4f}")
            print(f"Visual-Visual: mean={vv.mean():.4f}, std={vv.std():.4f}")
            print(f"Text-Visual:   mean={tv.mean():.4f}, std={tv.std():.4f}")
        
        return similarity_matrix
    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}
        # packed_text_embedding: text embedding len x 3584
        # past_key_values.key_cache[0]   past_key_values.value_cache[0]
        
        ########################  catch all hidden_states
        all_hidden_states = []
        hooks = []
        def make_hook_fn(layer_idx):
            """Create an independent hook callback for each layer."""
            def hook_fn(module, input, output):
                # The output may be a tuple; take the first element (usually hidden_state).
                if isinstance(output, tuple):
                    hidden_state = output[0]
                else:
                    hidden_state = output
                # Store as (layer_index, hidden_state).
                all_hidden_states.append((layer_idx, hidden_state.clone()))
            return hook_fn
        # Register hooks on every layer.
        layers = self.language_model.model.layers  # adjust to match the actual model structure
        for idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(make_hook_fn(idx))
            hooks.append(hook)    


        output = self.language_model.forward_inference( # 12x3584  11x3584
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            # output_hidden_states=True,
            # return_dict=True,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values


        ####################### Remove all hooks.
        for hook in hooks:
            hook.remove()
        # Sort by layer index (so the order is deterministic).
        all_hidden_states.sort(key=lambda x: x[0])
        # Drop the index and keep only the hidden_states.
        hidden_states_tuple_text = tuple(hs for _, hs in all_hidden_states) # hidden_states_tuple_text[0]:16x3584
        ANALYSIS_METHOD = 'NONE' # ATTN_SCORE   SVD  NOISE   ATTN_ENTROPY OVERSMOOTHING
        need_layer = 5 if ANALYSIS_METHOD=='NOISE' else 27
        remove_ids = [0, 1370, 1371, 1390]  # e.g. BOS at position 0, EOS at position 1390
        if past_key_values.value_cache[need_layer] is not None and True: # 1387x4x128   16x4x128
            if past_key_values.value_cache[need_layer].shape[0] > 1369:
                if ANALYSIS_METHOD == 'ATTN_SCORE':
                    for every_layer in range(28):#[6,10,14,18,22,26]:
                        hiddens = torch.cat([self.hidden_states_tuple_vit[every_layer], hidden_states_tuple_text[every_layer]], dim=0)
                        # attn_map = self.get_gqa_attention_map(hidden_states_tuple_text[every_layer], self.language_model.model.layers[every_layer].self_attn.q_proj, past_key_values.key_cache[every_layer], past_key_values.value_cache[every_layer])
                        # print(f'{every_layer} layer attn:{attn_map[:,:,:,1:1370].sum()}')
                        attn_map = self.compute_attention_map(
                            hiddens,  # [seq_len, hidden_size]
                            past_key_values.key_cache[every_layer],        # [seq_len, num_kv_heads, head_dim]
                            self.language_model.model.layers[every_layer].self_attn.q_proj.weight,
                            num_heads=28,
                            num_kv_heads=4,
                        )
                        attn_mean = attn_map.mean(dim=0)  # mean over all heads
                        attn_sum = attn_map.sum(dim=0)  # sum over all heads
                        results = self.analyze_attn_regions(attn_mean, vision_end=1371)
                        avg_t2v=results['avg_t2v']
                        print(f'{every_layer} layer attn:{avg_t2v:.4f}')
                elif ANALYSIS_METHOD == 'SVD':
                    for every_layer in range(28):#[6,10,14,18,22,26]:
                        hiddens = torch.cat([self.hidden_states_tuple_vit[every_layer], hidden_states_tuple_text[every_layer]], dim=0)
                        results = self.analyze_svd(self.hidden_states_tuple_vit[every_layer], layer_index=every_layer)
                elif ANALYSIS_METHOD == 'NOISE':
                    for every_layer in range(28):#[6,10,14,18,22,26]:
                        hiddens = torch.cat([self.hidden_states_tuple_vit[every_layer], hidden_states_tuple_text[every_layer]], dim=0)
                        attn_map = self.compute_attention_map(
                            hiddens,  # [seq_len, hidden_size]
                            past_key_values.key_cache[0],        # [seq_len, num_kv_heads, head_dim]
                            self.language_model.model.layers[0].self_attn.q_proj.weight,
                            num_heads=28,
                            num_kv_heads=4,
                        )                    
                elif ANALYSIS_METHOD == 'ATTN_ENTROPY':
                    for every_layer in range(28):#[6,10,14,18,22,26]:
                        hiddens = torch.cat([self.hidden_states_tuple_vit[every_layer], hidden_states_tuple_text[every_layer]], dim=0)
                        # attn_map = self.get_gqa_attention_map(hidden_states_tuple_text[every_layer], self.language_model.model.layers[every_layer].self_attn.q_proj, past_key_values.key_cache[every_layer], past_key_values.value_cache[every_layer])
                        # print(f'{every_layer} layer attn:{attn_map[:,:,:,1:1370].sum()}')
                        keep_ids = [i for i in range(hiddens.shape[0]) if i not in remove_ids]
                        attn_map = self.compute_attention_map(
                            hiddens[keep_ids],  # [seq_len, hidden_size]
                            past_key_values.key_cache[every_layer][keep_ids],        # [seq_len, num_kv_heads, head_dim]
                            self.language_model.model.layers[every_layer].self_attn.q_proj.weight,
                            num_heads=28,
                            num_kv_heads=4,
                        )
                        attn_mean = attn_map.mean(dim=0)  # mean over all heads
                        attn_sum = attn_map.sum(dim=0)  # sum over all heads
                        results = self.compute_attention_entropy(attn_mean, vision_end=1369, mode='t2v')
                        v2v_entropy, t2v_entropy = results
                        print(f"layer index:{every_layer}----V2V Entropy: {v2v_entropy:.4f} T2V Entropy: {t2v_entropy:.4f} ")
                        # print(f'{every_layer} layer attn:{avg_t2v:.4f}')             
                    print('done')      
                elif ANALYSIS_METHOD == 'OVERSMOOTHING':
                    for every_layer in [0,3,7,14,17,22,25,26]:#[6,10,14,18,22,26]:
                        hiddens = self.hidden_states_tuple_vit[every_layer]#torch.cat([self.hidden_states_tuple_vit[every_layer], hidden_states_tuple_text[every_layer]], dim=0)
                        save_name = 'analysis/cosine/reca_' + str(every_layer) + '.jpg'
                        hidden_states = hiddens.detach().cpu().to(torch.float32)

                        sim_matrix = self.visualize_token_similarity(hidden_states,save_path=save_name)
                else:
                    pass

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item() # packed_vit_tokens:1369x588 = 37*37x14*14*3
        # image-224x224   resize---> 518x518 = 14*37 patch_size = 14x14, patch_num = 37x37
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=packed_vit_tokens, 
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )# packed_vit_token_embed: 1369x1152  1152 is the dim of features
        packed_vit_token_embed = self.connector(packed_vit_token_embed) # 1152-->3584
        pos_emb = self.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed # 1369x3584

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        if past_key_values.value_cache[0] is not None and True:
            attn_map = self.get_gqa_attention_map(packed_text_embedding, self.language_model.model.layers[0].self_attn.q_proj, past_key_values.key_cache[0][1:1370], past_key_values.value_cache[0][1:1370])
            print(attn_map)


        ########################  catch all hidden_states
        all_hidden_states = []
        hooks = []
        def make_hook_fn(layer_idx):
            """Create an independent hook callback for each layer."""
            def hook_fn(module, input, output):
                # The output may be a tuple; take the first element (usually hidden_state).
                if isinstance(output, tuple):
                    hidden_state = output[0]
                else:
                    hidden_state = output
                # Store as (layer_index, hidden_state).
                all_hidden_states.append((layer_idx, hidden_state.clone()))
            return hook_fn
        # Register hooks on every layer.
        layers = self.language_model.model.layers  # adjust to match the actual model structure
        for idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(make_hook_fn(idx))
            hooks.append(hook)    

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values
        ####################### Remove all hooks.
        for hook in hooks:
            hook.remove()
        # Sort by layer index (so the order is deterministic).
        all_hidden_states.sort(key=lambda x: x[0])
        # Drop the index and keep only the hidden_states.
        hidden_states_tuple_vit = tuple(hs for _, hs in all_hidden_states)
        self.hidden_states_tuple_vit = hidden_states_tuple_vit
        
        ANALYSIS_METHOD = 'None' # ATTN_SCORE   SVD  NOISE
        need_layer = 5 if ANALYSIS_METHOD=='NOISE' else 27
        if past_key_values.value_cache[need_layer] is not None and True: # 1387x4x128   16x4x128
            if ANALYSIS_METHOD == 'NOISE':
                noised_layer = [0, 1, 2]  # tried [5,7,9,11,13], 22-27 did not work; max is 27  
                for noise_index in noised_layer:
                    value_cache = past_key_values.value_cache[noise_index]
                    zero_ratio = 1 - 0 / 701952  # zero-out 10% of the 701952 elements
                    print(f'layer:{noise_index}---zero ratio:{zero_ratio}')
                    mask = torch.rand_like(value_cache, dtype=torch.float32) > zero_ratio
                    value_cache_masked = value_cache * mask
                    past_key_values.value_cache[noise_index] = value_cache_masked

        return past_key_values


    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        # transform images
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input
    # The following helper methods are part of the Bagel class defined in bagel.py.
    def extract_keywords_from_prompt(
        self, 
        prompt: str, 
        tokenizer,
        custom_keywords: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract keywords from the prompt and locate their token positions.
        
        Args:
            prompt: input text prompt.
            tokenizer: tokenizer object.
            custom_keywords: optional user-specified keyword list.
        
        Returns:
            {
                'keywords': ['cat', 'red', ...],
                'keyword_token_indices': {
                    'cat': [12, 13],  # token indices for this keyword
                    'red': [8],
                    ...
                },
                'all_tokens': ['a', 'red', 'cat', ...],
                'num_tokens': int
            }
        """
        # Tokenize the prompt (without special tokens).
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        
        # Decide the keyword set.
        if custom_keywords:
            keywords = [kw.lower().strip() for kw in custom_keywords]
        else:
            # Simple heuristic: extract long-enough words and filter out stop words.
            words = prompt.lower().replace(',', ' ').replace('.', ' ').replace('"', ' ').split()
            stopwords = {
                'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 
                'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 
                'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'between',
                'under', 'again', 'further', 'then', 'once', 'here', 'there',
                'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
                'if', 'or', 'because', 'until', 'while', 'this', 'that', 'these',
                'those', 'what', 'which', 'who', 'whom', 'its', 'it', 'he', 'she',
                'they', 'them', 'his', 'her', 'their', 'my', 'your', 'our', 'like'
            }
            keywords = [w for w in words if w not in stopwords and len(w) > 2]
            # Deduplicate while preserving order.
            seen = set()
            keywords = [x for x in keywords if not (x in seen or seen.add(x))]
        
        # Locate each keyword inside the token list.
        keyword_token_indices = {}
        tokens_lower = [t.lower().strip() for t in tokens]
        
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            indices = []
            
            # Strategy 1: exact or partial match.
            for i, token in enumerate(tokens_lower):
                # Strip any special tokenizer characters (e.g. a leading Ġ for a space).
                token_clean = token.replace('Ġ', '').replace('▁', '').strip()
                if keyword_lower == token_clean or keyword_lower in token_clean or token_clean in keyword_lower:
                    if len(token_clean) > 1:  # avoid matching very short tokens
                        indices.append(i)
            
            # Strategy 2: fall back to subword matching if nothing was found.
            if not indices:
                keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
                keyword_token_strs = [tokenizer.decode([tid]).lower().strip() for tid in keyword_tokens]
                
                for i in range(len(tokens_lower) - len(keyword_token_strs) + 1):
                    match = True
                    for j, kt in enumerate(keyword_token_strs):
                        kt_clean = kt.replace('Ġ', '').replace('▁', '').strip()
                        token_clean = tokens_lower[i + j].replace('Ġ', '').replace('▁', '').strip()
                        if kt_clean not in token_clean and token_clean not in kt_clean:
                            match = False
                            break
                    if match:
                        indices.extend(range(i, i + len(keyword_token_strs)))
                        break
            
            if indices:
                keyword_token_indices[keyword] = list(set(indices))  # deduplicate
        
        result = {
            'keywords': list(keyword_token_indices.keys()),
            'keyword_token_indices': keyword_token_indices,
            'all_tokens': tokens,
            'token_ids': token_ids,
            'num_tokens': len(tokens),
        }
        
        # Log debug info.
        print(f"\n{'='*70}")
        print(f"KEYWORD EXTRACTION RESULTS")
        print(f"{'='*70}")
        print(f"Prompt: {prompt[:80]}...")
        print(f"Total tokens: {len(tokens)}")
        print(f"Keywords found: {len(keyword_token_indices)}")
        for kw, indices in keyword_token_indices.items():
            token_strs = [tokens[i] for i in indices if i < len(tokens)]
            print(f"  • '{kw}' -> indices {indices} -> tokens {token_strs}")
        print(f"{'='*70}\n")
        
        return result

        
    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # ============ Extra args for keyword attention analysis. ============
        analyze_attention: bool = False,
        analyze_timestep_indices: List[int] = None,
        analyze_layers: List[int] = None,
        keyword_info: Dict[str, Any] = None,
        text_token_range: Tuple[int, int] = None,
    ):
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        # Initialize keyword tracking.
        if analyze_attention:
            self._keyword_attention_tracking = {}
            if analyze_timestep_indices is None:
                # Default: analyse a few representative timesteps.
                total_steps = len(timesteps)
                analyze_timestep_indices = [0, total_steps // 4, total_steps // 2, total_steps - 1]
            if analyze_layers is None:
                analyze_layers = [0, 7, 14, 21, 27]
            
            print(f"\n{'='*70}")
            print(f"KEYWORD ATTENTION TRACKING ENABLED")
            print(f"Analyzing timestep indices: {analyze_timestep_indices}")
            print(f"Analyzing layers: {analyze_layers}")
            print(f"{'='*70}\n")

        for i, t in enumerate(timesteps):
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            
            # Decide whether to analyse the current step.
            should_analyze = (
                analyze_attention and 
                (i in analyze_timestep_indices) and 
                keyword_info is not None and
                text_token_range is not None
            )
            
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep, 
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                # ============ Forward analysis arguments. ============
                analyze_attention=should_analyze,
                analyze_layers=analyze_layers,
                keyword_info=keyword_info,
                text_token_range=text_token_range,
                current_timestep=t.item(),
                step_idx=i,
            )

            x_t = x_t - v_t.to(x_t.device) * dts[i]

        # Log a summary once generation has finished.
        if analyze_attention and hasattr(self, '_keyword_attention_tracking') and self._keyword_attention_tracking:
            self._print_keyword_attention_summary()

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # ============ Extra args for keyword attention analysis. ============
        analyze_attention: bool = False,
        analyze_layers: List[int] = None,
        keyword_info: Dict[str, Any] = None,
        text_token_range: Tuple[int, int] = None,
        current_timestep: float = None,
        step_idx: int = None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)
        x_t_embed = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        packed_sequence[packed_vae_token_indexes] = x_t_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        
        # ============ Analyse keyword attention (using the manual implementation). ============
        if analyze_attention and keyword_info and text_token_range and analyze_layers:
            # Initialize the tracking buffers.
            if not hasattr(self, '_keyword_attention_tracking'):
                self._keyword_attention_tracking = {}
            if current_timestep not in self._keyword_attention_tracking:
                self._keyword_attention_tracking[current_timestep] = {}
            
            # Indices of latent tokens in the query.
            latent_indices = packed_vae_token_indexes.tolist()
            
            # Offset of text tokens in the KV cache.
            text_start, text_end = text_token_range
            
            for layer_idx in analyze_layers:
                if layer_idx >= len(self.language_model.model.layers):
                    continue
                
                # Check whether the KV cache is valid.
                if past_key_values.key_cache[layer_idx] is None:
                    continue
                
                try:
                    # Manually compute the attention map.
                    attn_map = self.compute_attention_map(
                        packed_sequence,
                        past_key_values.key_cache[layer_idx],
                        self.language_model.model.layers[layer_idx].self_attn.q_proj.weight,
                        num_heads=self.num_heads,
                        num_kv_heads=self.language_model.config.num_key_value_heads,
                    )
                    # attn_map shape: [num_heads, query_len, kv_len]
                    
                    layer_results = self.analyze_keyword_attention(
                        attn_weights=attn_map,
                        latent_token_indices=latent_indices,
                        keyword_token_indices=keyword_info['keyword_token_indices'],
                        text_token_offset_in_kv=text_start,
                        timestep=current_timestep,
                        layer_idx=layer_idx,
                    )
                    
                    if layer_results:
                        self._keyword_attention_tracking[current_timestep][layer_idx] = layer_results
                        
                        # Log the current result.
                        print(f"[Step {step_idx}][t={current_timestep:.4f}][Layer {layer_idx}] Keyword Attention:")
                        for kw, attn_pct in layer_results['keyword_attention_pct'].items():
                            print(f"    • '{kw}': {attn_pct:.2f}%")
                
                except Exception as e:
                    print(f"[Warning] Failed to analyze attention at layer {layer_idx}: {e}")
                    continue

        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        # ============ CFG handling (unchanged logic). ============
        if cfg_text_scale > 1.0:
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not supported")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                v_t = v_t_ * scale

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def analyze_keyword_attention(
        self,
        attn_weights: torch.Tensor,           # [num_heads, query_len, kv_len]
        latent_token_indices: List[int],       # indices of latent tokens in the query
        keyword_token_indices: Dict[str, List[int]],  # keyword -> token indices in KV
        text_token_offset_in_kv: int,          # offset of text tokens in the KV cache
        timestep: float = None,
        layer_idx: int = None,
    ) -> Dict[str, Any]:
        """
        Analyse how much attention each keyword receives.
        
        Args:
            attn_weights: attention weights, shape [num_heads, query_len, kv_len]
            latent_token_indices: indices of latent tokens inside the query sequence.
            keyword_token_indices: keyword -> token indices inside the original prompt.
            text_token_offset_in_kv: starting offset of text tokens in the KV cache.
            timestep: current denoising timestep.
            layer_idx: layer index.
        
        Returns:
            A dictionary of summary statistics.
        """
        num_heads, query_len, kv_len = attn_weights.shape
        
        # Extract the attention that corresponds to latent tokens.
        # ``latent_token_indices`` are latent-token positions in the query.
        if len(latent_token_indices) == 0:
            return {}
        
        # Extract latent-token attention with shape [num_heads, num_latent, kv_len].
        latent_attn = attn_weights[:, latent_token_indices, :]
        
        # Mean over heads -> [num_latent, kv_len].
        latent_attn_mean = latent_attn.mean(dim=0)
        
        # Mean over all latent tokens -> per-KV-position average attention [kv_len].
        attn_per_kv = latent_attn_mean.mean(dim=0)
        
        # Compute the attention received by each keyword.
        keyword_attention = {}
        keyword_attention_sum = {}
        
        for keyword, indices in keyword_token_indices.items():
            # Convert to actual indices in the KV cache.
            kv_indices = [i + text_token_offset_in_kv for i in indices]
            # Keep only the indices within the valid range.
            valid_kv_indices = [i for i in kv_indices if 0 <= i < kv_len]
            
            if valid_kv_indices:
                kw_attn_values = attn_per_kv[valid_kv_indices]
                keyword_attention_sum[keyword] = kw_attn_values.sum().item()
                keyword_attention[keyword] = kw_attn_values.mean().item()
        
        # Compute total attention (over all text tokens).
        # Assume text tokens start at text_token_offset_in_kv.
        total_attention = attn_per_kv.sum().item()
        
        # Normalize: fraction of total attention that each keyword takes (%).
        keyword_attention_pct = {
            kw: (attn_sum / total_attention * 100) if total_attention > 0 else 0 
            for kw, attn_sum in keyword_attention_sum.items()
        }
        
        results = {
            'keyword_attention_mean': keyword_attention,
            'keyword_attention_sum': keyword_attention_sum,
            'keyword_attention_pct': keyword_attention_pct,
            'total_attention': total_attention,
        }
        
        return results


    def _print_keyword_attention_summary(self):
        """Log a summary of the keyword attention tracking."""
        if not hasattr(self, '_keyword_attention_tracking') or not self._keyword_attention_tracking:
            print("No keyword attention tracking results available.")
            return
        
        print("\n" + "=" * 100)
        print("KEYWORD ATTENTION TRACKING SUMMARY")
        print("=" * 100)
        
        # Collect all tracked timesteps and layers.
        all_timesteps = sorted(self._keyword_attention_tracking.keys(), reverse=True)
        all_layers = set()
        all_keywords = set()
        
        for t_data in self._keyword_attention_tracking.values():
            for layer_idx, layer_data in t_data.items():
                all_layers.add(layer_idx)
                all_keywords.update(layer_data['keyword_attention_pct'].keys())
        
        all_layers = sorted(all_layers)
        all_keywords = sorted(all_keywords)
        
        if not all_keywords:
            print("No keywords tracked.")
            return
        
        # Print a table for each layer.
        for layer_idx in all_layers:
            print(f"\n📊 Layer {layer_idx} - Keyword Attention (% of total)")
            print("-" * (15 + 12 * len(all_keywords)))
            
            # Header row.
            header = f"{'Timestep':<15}"
            for kw in all_keywords:
                header += f"{kw[:10]:<12}"
            print(header)
            print("-" * (15 + 12 * len(all_keywords)))
            
            # One row per timestep.
            for t in all_timesteps:
                if layer_idx in self._keyword_attention_tracking.get(t, {}):
                    row = f"{t:<15.4f}"
                    kw_attn = self._keyword_attention_tracking[t][layer_idx]['keyword_attention_pct']
                    for kw in all_keywords:
                        val = kw_attn.get(kw, 0)
                        row += f"{val:<12.2f}"
                    print(row)
        
        # Compute and log the average attention for each keyword.
        print("\n" + "=" * 100)
        print("KEYWORD ATTENTION TRENDS (averaged across layers and timesteps)")
        print("=" * 100)
        
        keyword_avg = {kw: [] for kw in all_keywords}
        
        for t in all_timesteps:
            for layer_idx in all_layers:
                if layer_idx in self._keyword_attention_tracking.get(t, {}):
                    kw_attn = self._keyword_attention_tracking[t][layer_idx]['keyword_attention_pct']
                    for kw in all_keywords:
                        if kw in kw_attn:
                            keyword_avg[kw].append(kw_attn[kw])
        
        # Sort by average attention.
        keyword_stats = []
        for kw in all_keywords:
            values = keyword_avg[kw]
            if values:
                avg = sum(values) / len(values)
                keyword_stats.append((kw, avg, min(values), max(values)))
        
        keyword_stats.sort(key=lambda x: x[1], reverse=True)
        
        for kw, avg, min_val, max_val in keyword_stats:
            print(f"  • '{kw}': avg={avg:.2f}%, range=[{min_val:.2f}%, {max_val:.2f}%]")
        
        # Store the data for later plotting.
        self._keyword_summary = {
            'timesteps': all_timesteps,
            'layers': all_layers,
            'keywords': all_keywords,
            'keyword_stats': keyword_stats,
        }
        
        print("=" * 100)


    def plot_keyword_attention_curves(
        self,
        save_path: str = None,
        figsize: Tuple[int, int] = (14, 6),
    ):
        """
        Plot curves showing how keyword attention evolves over timesteps.
        """
        if not hasattr(self, '_keyword_attention_tracking') or not self._keyword_attention_tracking:
            print("No keyword attention tracking results available.")
            return
        
        all_timesteps = sorted(self._keyword_attention_tracking.keys(), reverse=True)
        all_layers = set()
        all_keywords = set()
        
        for t_data in self._keyword_attention_tracking.values():
            for layer_idx, layer_data in t_data.items():
                all_layers.add(layer_idx)
                all_keywords.update(layer_data['keyword_attention_pct'].keys())
        
        all_layers = sorted(all_layers)
        all_keywords = sorted(all_keywords)
        
        if not all_keywords:
            print("No keywords to plot.")
            return
        
        # Average attention per timestep (averaged across layers).
        keyword_trends = {kw: [] for kw in all_keywords}
        
        for t in all_timesteps:
            for kw in all_keywords:
                layer_values = []
                for layer_idx in all_layers:
                    if layer_idx in self._keyword_attention_tracking.get(t, {}):
                        val = self._keyword_attention_tracking[t][layer_idx]['keyword_attention_pct'].get(kw, 0)
                        layer_values.append(val)
                avg_val = sum(layer_values) / len(layer_values) if layer_values else 0
                keyword_trends[kw].append(avg_val)
        
        # Build the color map.
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_keywords)))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left subplot: attention over time.
        x_positions = range(len(all_timesteps))
        for i, kw in enumerate(all_keywords):
            ax1.plot(x_positions, keyword_trends[kw], 
                    marker='o', label=kw, color=colors[i], linewidth=2, markersize=6)
        
        ax1.set_xlabel('Denoising Step (t: 1.0 → 0.0)', fontsize=12)
        ax1.set_ylabel('Attention (%)', fontsize=12)
        ax1.set_title('Keyword Attention During Generation', fontsize=14)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels([f"{t:.2f}" for t in all_timesteps], rotation=45, fontsize=9)
        
        # Right subplot: average attention per keyword (bar chart).
        avg_attention = {kw: np.mean(keyword_trends[kw]) for kw in all_keywords}
        sorted_keywords = sorted(avg_attention.items(), key=lambda x: x[1], reverse=True)
        
        kw_names = [x[0] for x in sorted_keywords]
        kw_values = [x[1] for x in sorted_keywords]
        kw_colors = [colors[all_keywords.index(kw)] for kw in kw_names]
        
        bars = ax2.barh(range(len(kw_names)), kw_values, color=kw_colors)
        ax2.set_yticks(range(len(kw_names)))
        ax2.set_yticklabels(kw_names, fontsize=10)
        ax2.set_xlabel('Average Attention (%)', fontsize=12)
        ax2.set_title('Average Keyword Attention', fontsize=14)
        ax2.invert_yaxis()
        
        # Add numeric labels.
        for bar, val in zip(bars, kw_values):
            ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                    f'{val:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved keyword attention plot to: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)


    def plot_keyword_attention_heatmap(
        self,
        save_path: str = None,
        figsize: Tuple[int, int] = (14, 8),
    ):
        """
        Plot a heatmap of keyword attention (timestep x keyword).
        """
        if not hasattr(self, '_keyword_attention_tracking') or not self._keyword_attention_tracking:
            print("No keyword attention tracking results available.")
            return
        
        all_timesteps = sorted(self._keyword_attention_tracking.keys(), reverse=True)
        all_layers = sorted(set(
            layer for t_data in self._keyword_attention_tracking.values() 
            for layer in t_data.keys()
        ))
        all_keywords = sorted(set(
            kw for t_data in self._keyword_attention_tracking.values() 
            for layer_data in t_data.values() 
            for kw in layer_data['keyword_attention_pct'].keys()
        ))
        
        if not all_keywords:
            print("No keywords to plot.")
            return
        
        n_layers = len(all_layers)
        fig, axes = plt.subplots(1, n_layers, figsize=(figsize[0], figsize[1]))
        if n_layers == 1:
            axes = [axes]
        
        for ax, layer_idx in zip(axes, all_layers):
            # Build the heatmap data for this layer.
            data = np.zeros((len(all_timesteps), len(all_keywords)))
            
            for i, t in enumerate(all_timesteps):
                if layer_idx in self._keyword_attention_tracking.get(t, {}):
                    kw_attn = self._keyword_attention_tracking[t][layer_idx]['keyword_attention_pct']
                    for j, kw in enumerate(all_keywords):
                        data[i, j] = kw_attn.get(kw, 0)
            
            im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
            ax.set_xticks(range(len(all_keywords)))
            ax.set_xticklabels(all_keywords, rotation=45, ha='right', fontsize=9)
            ax.set_yticks(range(len(all_timesteps)))
            ax.set_yticklabels([f"{t:.2f}" for t in all_timesteps], fontsize=9)
            ax.set_xlabel('Keyword', fontsize=10)
            ax.set_ylabel('Timestep', fontsize=10)
            ax.set_title(f'Layer {layer_idx}', fontsize=11)
            
            # Overlay numeric values.
            for i in range(len(all_timesteps)):
                for j in range(len(all_keywords)):
                    ax.text(j, i, f'{data[i,j]:.1f}', ha='center', va='center', fontsize=7,
                        color='white' if data[i,j] > data.max()/2 else 'black')
        
        plt.colorbar(im, ax=axes, label='Attention (%)', shrink=0.8)
        fig.suptitle('Keyword Attention Heatmap by Layer', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved keyword attention heatmap to: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]

        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope, 
                images=[image], 
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            prompts=[prompt],
            tokenizer=tokenizer, 
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output