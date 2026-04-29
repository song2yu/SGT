# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt

from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache


VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        self.tSNE = False  # Must be False here -- otherwise the call returns early.

    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        if vae:
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

            if self.tSNE:
                cu_seqlens = torch.nn.functional.pad(torch.cumsum(generation_input['vit_token_seqlens'], dim=0), (1, 0))
                cu_seqlens = cu_seqlens.to(torch.int32)
                max_seqlen = torch.max(generation_input['vit_token_seqlens']).item()
                packed_vit_token_embed = self.model.vit_model(
                    packed_pixel_values=generation_input['packed_vit_tokens'], 
                    packed_flattened_position_ids=generation_input['packed_vit_position_ids'],
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                return packed_vit_token_embed

        gen_context['kv_lens'] = kv_lens 
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50, 
        timestep_shift=3.0,
        # ============ Keyword analysis arguments. ============
        analyze_attention: bool = False,
        analyze_timestep_indices: List[int] = None,
        analyze_layers: List[int] = None,
        keyword_info: Dict[str, Any] = None,
        text_token_range: Tuple[int, int] = None,
    ):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            # ============ Forward the keyword analysis arguments. ============
            analyze_attention=analyze_attention,
            analyze_timestep_indices=analyze_timestep_indices,
            analyze_layers=analyze_layers,
            keyword_info=keyword_info,
            text_token_range=text_token_range,
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        image_understanding_to_image=False,
        image_shapes=(512, 512),
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        # ============ Keyword analysis arguments. ============
        analyze_attention: bool = False,
        analyze_timestep_indices: List[int] = None,
        analyze_layers: List[int] = None,
        custom_keywords: List[str] = None,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        # ============ Track keyword info and text-span positions. ============
        keyword_info = None
        text_token_start = 0
        text_token_end = 0
        full_prompt = ""

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    # Record the start position of the text tokens.
                    text_token_start = gen_context['kv_lens'][0]
                    
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)
                    
                    # Record the end position of the text tokens.
                    text_token_end = gen_context['kv_lens'][0]
                    full_prompt += input_term + " "
                    
                    if analyze_attention:
                        print(f"[INFO] Text processed: '{input_term[:60]}...'")
                        print(f"[INFO] Text token range in KV cache: ({text_token_start}, {text_token_end})")

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    
                    if image_understanding_to_image:
                        gen_context = self.update_context_image(input_term, gen_context, vae=False, vit=True)
                    else:
                        gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)
                        if self.tSNE:
                            return gen_context

                    if image_shapes is None:
                        image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            # ============ Extract keywords. ============
            if analyze_attention and full_prompt.strip():
                keyword_info = self.model.extract_keywords_from_prompt(
                    prompt=full_prompt.strip(),
                    tokenizer=self.tokenizer,
                    custom_keywords=custom_keywords,
                )

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature)
                output_list.append(gen_text)
            else:
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)
                    text_token_end = gen_context['kv_lens'][0]
                
                print(f'image_shape: {image_shapes}')
                
                # Build the text-token span.
                text_token_range = (text_token_start, text_token_end) if text_token_end > text_token_start else None
                
                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    # ============ Forward the keyword analysis arguments. ============
                    analyze_attention=analyze_attention,
                    analyze_timestep_indices=analyze_timestep_indices,
                    analyze_layers=analyze_layers,
                    keyword_info=keyword_info,
                    text_token_range=text_token_range,
                )

                output_list.append(img)

        return output_list

    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        image_understanding_to_image: bool = False,
        # ============ Keyword analysis arguments. ============
        analyze_attention: bool = False,
        analyze_timestep_indices: List[int] = None,
        analyze_layers: List[int] = None,
        custom_keywords: List[str] = None,
        **kargs
    ) -> Dict[str, Any]:
        """
        Main entry point.
        
        Args:
            image: optional input image.
            text: optional input text.
            image_understanding_to_image: whether to enable image-understanding-to-image generation.
            analyze_attention: whether to analyse keyword attention.
            analyze_timestep_indices: list of timestep indices to analyse.
            analyze_layers: list of layer indices to analyse.
            custom_keywords: user-specified keyword list.
            **kargs: additional arguments forwarded to interleave_inference.
        
        Returns:
            A dict containing the "image" and "text" entries.
        """
        output_dict = {'image': None, 'text': None}

        if image is None and image_understanding_to_image:
            print('For image_understanding_to_image, an image input is required.')
            return output_dict

        if image_understanding_to_image and text is None:
            text = "Describe the Image"
            print(f"No prompt provided for image understanding. Using default: '{text}'")

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(
            input_list, 
            image_understanding_to_image=image_understanding_to_image,
            analyze_attention=analyze_attention,
            analyze_timestep_indices=analyze_timestep_indices,
            analyze_layers=analyze_layers,
            custom_keywords=custom_keywords,
            **kargs
        )
        
        if self.tSNE:
            return output_list
        
        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        
        return output_dict