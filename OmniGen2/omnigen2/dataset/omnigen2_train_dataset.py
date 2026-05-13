from typing import Optional, Union, List

import os
import random
import yaml
import copy
import pdb
import glob
from PIL import Image
from .consts import get_segment_prompt_list, get_recon_prompt_list, get_edge_prompt_list
import torch
# from torchvision import transforms
from transformers import AutoProcessor

from datasets import load_dataset, concatenate_datasets

from ..pipelines.omnigen2.pipeline_omnigen2 import OmniGen2ImageProcessor

class OmniGen2TrainDataset(torch.utils.data.Dataset):
    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."
    SYSTEM_PROMPT_DROP = "You are a helpful assistant that generates images."

    def __init__(
        self,
        config_path: str,
        tokenizer,
        use_chat_template: bool,
        max_input_pixels: Optional[Union[int, List[int]]] = None,
        max_output_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
        img_scale_num: int = 16,
        prompt_dropout_prob: float = 0.0,
        ref_img_dropout_prob: float = 0.0,
    ):
        self.max_input_pixels = max_input_pixels
        self.max_output_pixels = max_output_pixels

        self.max_side_length = max_side_length
        self.img_scale_num = img_scale_num
        self.prompt_dropout_prob = prompt_dropout_prob
        self.ref_img_dropout_prob = ref_img_dropout_prob

        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

        self.use_chat_template = use_chat_template
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=img_scale_num, do_resize=True)

        data = self._collect_annotations(self.config)

        self.data = data
        self.tokenizer = tokenizer

        # --- Pre-compute task-type partitions -----------------------------
        # ``TaskTypeDistributedSampler`` uses these to stream micro-batches
        # in either "all same type" or "interleave" mode.
        # An item is SFT iff it carries an ``id`` (LLaVA-OneVision schema);
        # everything else is treated as a generation-task sample.
        self.sft_indices: List[int] = []
        self.gen_indices: List[int] = []
        for i, item in enumerate(self.data):
            item_id = item.get('id') if isinstance(item, dict) else None
            if item_id not in (None, ''):
                self.sft_indices.append(i)
            else:
                self.gen_indices.append(i)
        # Data locations. Override via env vars to make the code portable.
        import os
        self.data_base = os.environ.get('OMNIGEN2_OMNIEDIT_ROOT', 'data/omniedit/images/')
        self.coco_base = os.environ.get('OMNIGEN2_COCO_ROOT', 'data/train2017/')
        self.sft_base = os.environ.get('OMNIGEN2_SFT_IMAGE_ROOT', 'data/llava_onevision/images')
        # Root for SAM-SGT images. Each txt entry is either an absolute path
        # under ``sam_selection/`` or a path relative to this root. The mask
        # is obtained by swapping ``sam_selection`` -> ``sam_mask`` and the
        # extension from .jpg to .png.
        self.sam_base = os.environ.get('OMNIGEN2_SAM_ROOT', 'data/SAM-SGT')
        self.get_recon_prompt_list = get_recon_prompt_list()
        self.get_segment_prompt_list = get_segment_prompt_list()
        self.get_edge_prompt_list = get_edge_prompt_list()
        # Prefer a local processor dir populated by shells/download_pretrained.sh
        # to avoid hitting HuggingFace Hub at every process start.
        qwen_processor_path = os.environ.get(
            'OMNIGEN2_QWEN_PROCESSOR_PATH',
            'Qwen/Qwen2.5-VL-3B-Instruct',
        )
        self.processor = AutoProcessor.from_pretrained(
            qwen_processor_path,
            min_pixels=256 * 28 * 28,      # Minimum number of pixels
            max_pixels=1280 * 28 * 28,     # Maximum number of pixels (about 1280 * 1280)
        )

    def _resolve_data_root(self, path: str) -> str:
        """Replace the ``<DATA_ROOT>`` placeholder in a config path with the
        value of the ``OMNIGEN2_DATA_ROOT`` env var (falls back to ``data``).
        Keeps the rest of the string untouched so existing absolute paths
        still work unchanged.
        """
        if not isinstance(path, str) or "<DATA_ROOT>" not in path:
            return path
        data_root = os.environ.get("OMNIGEN2_DATA_ROOT", "data")
        return path.replace("<DATA_ROOT>", data_root.rstrip("/"))

    def _collect_annotations(self, config):
        total_samples = 0
        total_ratio = 0
        json_datasets = []
        for data in config['data']:
            data_path, data_type = data['path'], data.get("type", "default")
            data_path = self._resolve_data_root(data_path)
            if os.path.isdir(data_path):
                jsonl_files = list(glob.glob(os.path.join(data_path, "**/*.jsonl"), recursive=True)) + list(glob.glob(os.path.join(data_path, "**/*.json"), recursive=True))
                json_dataset = load_dataset('json', data_files=jsonl_files, cache_dir=None)['train']
            else:
                data_ext = os.path.splitext(data_path)[-1]
                
                if data_ext in [".json", ".jsonl"]:
                    json_dataset = load_dataset('json', data_files=data_path, cache_dir=None)['train']
                elif data_ext in [".yml", ".yaml"]:
                    with open(data_path, "r") as f:
                        sub_config = yaml.load(f, Loader=yaml.FullLoader)
                        json_dataset = self._collect_annotations(sub_config)
                elif data_ext in [".txt"]:
                    json_dataset = load_dataset('text', data_files=data_path, cache_dir=None)['train']
                else:
                    raise NotImplementedError(
                        f'Unknown data file extension: "{data_ext}". '
                        f"Currently, .json, .jsonl .yml .yaml are supported. "
                        "If you are using a supported format, please set the file extension so that the proper parsing "
                        "routine can be called."
                    )
            total_ratio += data['ratio']
            total_samples += len(json_dataset)
            json_datasets.append(json_dataset)

        for json_dataset in json_datasets:
            
            target_size = int(len(json_dataset) * data['ratio'] / total_ratio) # normalize the ratio
            if target_size <= len(json_dataset):
                # Random selection without replacement
                indices = random.sample(range(len(json_dataset)), target_size)
            else:
                # Oversample with replacement
                indices = random.choices(range(len(json_dataset)), k=target_size)
            json_dataset = json_dataset.select(indices)
            
        json_dataset = concatenate_datasets(json_datasets)
        return json_dataset
    
    def clean_data_item(self, data_item):
        task_type = data_item['task_type']
        prefixs = ["The image portrays ", "The image depicts ", "The image captures ", "The image highlights ", "The image shows "]
        if "text_to_image" in task_type or "t2i" in task_type:
            if random.random() < 0.5:
                for p in prefixs:
                    if p in data_item['spatial_instruction']:
                        data_item['spatial_instruction'] = data_item['spatial_instruction'].replace(p, "")
                        break
        return data_item
    
    def apply_chat_template(self, instruction, system_prompt, ues_reca=True):
        # if self.use_chat_template:
        #     prompt = [
        #         {
        #             "role": "system",
        #             "content": system_prompt,
        #         },
        #         {"role": "user", "content": instruction},
        #     ]
        #     instruction = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        if ues_reca: # images is not None
            prompt = "".join(
                [
                    f"<img{i}>: <|vision_start|><|image_pad|><|vision_end|>"
                    for i in range(1, 2)
                ]
            ) + instruction
        instruction = f"<|im_start|>system\nYou are a helpful assistant that generates high-quality images based on user instructions.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        return instruction
    
    def process_item(self, data_item):
        # assert data_item['instruction'] is not None
        data_item['task_type'] = 't2i'
        data_item = self.clean_data_item(data_item)

        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob

        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            instruction = self.apply_chat_template(data_item['spatial_instruction'], self.SYSTEM_PROMPT)

        if not drop_ref_img and 'omni_edit_id' in data_item and data_item['omni_edit_id'] is not None:
            input_file = data_item['omni_edit_id'] + '_src.png'
            output_file = data_item['omni_edit_id'] + '_edited.png'
            input_images_path = [os.path.join(self.data_base, input_file)]
            input_images = []

            max_input_pixels = self.max_input_pixels[0]# self.max_input_pixels[len(input_images_path) - 1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels

            for input_image_path in input_images_path:
                input_image = Image.open(input_image_path).convert("RGB")
                input_image = self.image_processor.preprocess(input_image, max_pixels=max_input_pixels, max_side_length=self.max_side_length)
                input_images.append(input_image)
        else:
            input_images_path, input_images = None, None

        output_image_path = os.path.join(self.data_base, output_file) # data_item['output_image']
        output_image = Image.open(output_image_path).convert("RGB")
        output_image = self.image_processor.preprocess(output_image, max_pixels=self.max_output_pixels, max_side_length=self.max_side_length)

        data = {
            'task_type': data_item['task'],
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': output_image_path,
        }
        return data

    def process_reca(self, data_item):
        # assert data_item['instruction'] is not None
        data_item['task_type'] = 'reca'
        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob

        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            prompt = random.choice(self.get_recon_prompt_list)
            instruction = self.apply_chat_template(prompt, self.SYSTEM_PROMPT)

        if not drop_ref_img:
            input_file = data_item['text']
            output_file = input_file#.replace()
            input_images_path = [os.path.join(self.coco_base, input_file)]
            input_images = []

            max_input_pixels = self.max_input_pixels[0]# self.max_input_pixels[len(input_images_path) - 1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels

            for input_image_path in input_images_path:
                input_image = Image.open(input_image_path).convert("RGB")
                input_image = self.image_processor.preprocess(input_image, max_pixels=max_input_pixels, max_side_length=self.max_side_length)
                input_images.append(input_image)

            # Temporarily switch padding side (as in the reference implementation)
            original_padding_side = self.processor.tokenizer.padding_side
            self.processor.tokenizer.padding_side = "left" # or "right"; for an encoder-only feature extractor, "right" also works

            inputs_items = self.processor(
                text=instruction,
                images=input_images_path,
                padding=True,
                return_tensors="pt",
            )
            # Restore padding side
            self.processor.tokenizer.padding_side = original_padding_side
        else:
            input_images_path, input_images = None, None
        # pdb.set_trace()
        output_image_path = os.path.join(self.coco_base, output_file) # data_item['output_image']
        output_image = Image.open(output_image_path).convert("RGB")
        output_image = self.image_processor.preprocess(output_image, max_pixels=self.max_output_pixels, max_side_length=self.max_side_length)
        data = {
            'task_type': data_item['task_type'],
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': output_image_path,
            'input_ids':inputs_items.input_ids,
            'attention_mask':inputs_items.attention_mask,
            'pixel_values':inputs_items['pixel_values'],
            'image_grid_thw':inputs_items['image_grid_thw']
        }

        return data

    def process_edge(self, data_item):
        # assert data_item['instruction'] is not None
        data_item['task_type'] = 'edge'
        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob

        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            prompt = random.choice(self.get_edge_prompt_list)
            instruction = self.apply_chat_template(prompt, self.SYSTEM_PROMPT)

        if not drop_ref_img:
            input_file = data_item['text']
            input_images_path = [os.path.join(self.coco_base, input_file)]
            output_file = input_images_path[0].replace('train2017', 'coco_edge').replace('jpg', 'png') 
            input_images = []

            max_input_pixels = self.max_input_pixels[0]# self.max_input_pixels[len(input_images_path) - 1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels

            for input_image_path in input_images_path:
                input_image = Image.open(input_image_path).convert("RGB")
                input_image = self.image_processor.preprocess(input_image, max_pixels=max_input_pixels, max_side_length=self.max_side_length)
                input_images.append(input_image)

            # Temporarily switch padding side (as in the reference implementation)
            original_padding_side = self.processor.tokenizer.padding_side
            self.processor.tokenizer.padding_side = "left" # or "right"; for an encoder-only feature extractor, "right" also works

            inputs_items = self.processor(
                text=instruction,
                images=input_images_path,
                padding=True,
                return_tensors="pt",
            )
            # Restore padding side
            self.processor.tokenizer.padding_side = original_padding_side
        else:
            input_images_path, input_images = None, None

        output_image_path = os.path.join(self.coco_base, output_file) # data_item['output_image']
        output_image = Image.open(output_image_path).convert("RGB")
        output_image = self.image_processor.preprocess(output_image, max_pixels=self.max_output_pixels, max_side_length=self.max_side_length)

        data = {
            'task_type': data_item['task_type'],
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': output_image_path,
            'input_ids':inputs_items.input_ids,
            'attention_mask':inputs_items.attention_mask,
            'pixel_values':inputs_items['pixel_values'],
            'image_grid_thw':inputs_items['image_grid_thw']
        }
        return data

    def process_semantic(self, data_item):
        # assert data_item['instruction'] is not None
        data_item['task_type'] = 'semantic'
        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob

        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            prompt = random.choice(self.get_segment_prompt_list)
            instruction = self.apply_chat_template(prompt, self.SYSTEM_PROMPT)
            
        if not drop_ref_img:
            input_file = data_item['text']
            input_images_path = [os.path.join(self.coco_base, input_file)]
            output_file = input_images_path[0].replace('train2017', 'annotations/stuff_train2017_pixelmaps').replace('jpg', 'png') 
            input_images = []

            max_input_pixels = self.max_input_pixels[0]# self.max_input_pixels[len(input_images_path) - 1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels

            for input_image_path in input_images_path:
                input_image = Image.open(input_image_path).convert("RGB")
                input_image = self.image_processor.preprocess(input_image, max_pixels=max_input_pixels, max_side_length=self.max_side_length)
                input_images.append(input_image)

            # Temporarily switch padding side (as in the reference implementation)
            original_padding_side = self.processor.tokenizer.padding_side
            self.processor.tokenizer.padding_side = "left" # or "right"; for an encoder-only feature extractor, "right" also works

            inputs_items = self.processor(
                text=instruction,
                images=input_images_path,
                padding=True,
                return_tensors="pt",
            )
            # Restore padding side
            self.processor.tokenizer.padding_side = original_padding_side
        else:
            input_images_path, input_images = None, None

        output_image_path = os.path.join(self.coco_base, output_file) # data_item['output_image']
        output_image = Image.open(output_image_path).convert("RGB")
        output_image = self.image_processor.preprocess(output_image, max_pixels=self.max_output_pixels, max_side_length=self.max_side_length)

        data = {
            'task_type': data_item['task_type'],
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': output_image_path,
            'input_ids':inputs_items.input_ids,
            'attention_mask':inputs_items.attention_mask,
            'pixel_values':inputs_items['pixel_values'],
            'image_grid_thw':inputs_items['image_grid_thw']
        }
        return data

    def process_panoptic(self, data_item):
        # assert data_item['instruction'] is not None
        data_item['task_type'] = 'panoptic'
        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob

        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            prompt = random.choice(self.get_segment_prompt_list)
            instruction = self.apply_chat_template(prompt, self.SYSTEM_PROMPT)
            
        if not drop_ref_img:
            input_file = data_item['text']
            input_images_path = [os.path.join(self.coco_base, input_file)]
            output_file = input_images_path[0].replace('train2017', 'annotations/panoptic_train2017').replace('jpg', 'png')
            input_images = []

            max_input_pixels = self.max_input_pixels[0]# self.max_input_pixels[len(input_images_path) - 1] if isinstance(self.max_input_pixels, list) else self.max_input_pixels

            for input_image_path in input_images_path:
                input_image = Image.open(input_image_path).convert("RGB")
                input_image = self.image_processor.preprocess(input_image, max_pixels=max_input_pixels, max_side_length=self.max_side_length)
                input_images.append(input_image)

            # Temporarily switch padding side (as in the reference implementation)
            original_padding_side = self.processor.tokenizer.padding_side
            self.processor.tokenizer.padding_side = "left" # or "right"; for an encoder-only feature extractor, "right" also works

            inputs_items = self.processor(
                text=instruction,
                images=input_images_path,
                padding=True,
                return_tensors="pt",
            )
            # Restore padding side
            self.processor.tokenizer.padding_side = original_padding_side
        else:
            input_images_path, input_images = None, None

        output_image_path = os.path.join(self.coco_base, output_file) # data_item['output_image']
        output_image = Image.open(output_image_path).convert("RGB")
        output_image = self.image_processor.preprocess(output_image, max_pixels=self.max_output_pixels, max_side_length=self.max_side_length)
        data = {
            'task_type': data_item['task_type'],
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': output_image_path,
            'input_ids':inputs_items.input_ids,
            'attention_mask':inputs_items.attention_mask,
            'pixel_values':inputs_items['pixel_values'],
            'image_grid_thw':inputs_items['image_grid_thw']
        }
        return data

    def process_sam(self, data_item):
        """Process a SAM-SGT item (image -> segmentation mask).

        Each ``data_item`` comes from the SAM txt list and therefore looks
        like ``{'text': '<path to sam_selection/XYZ.jpg>'}``. Following
        BAGEL/data/sam_190k.py:

        * The input image lives under ``sam_selection/``.
        * The paired mask is obtained by replacing ``sam_selection`` with
          ``sam_mask`` and changing the extension from ``.jpg`` to ``.png``.
        * The instruction is randomly sampled from the segmentation prompt
          list (shared with panoptic / semantic).

        The resulting dict mirrors the shape returned by
        ``process_panoptic`` so it flows through the existing collator
        without further changes.
        """
        data_item['task_type'] = 'sam'

        # Keep SAM deterministic: never drop the prompt / reference image.
        # (Random dropout was useful for pure generation tasks but here we
        # want every sample to actually contribute the segmentation signal.)
        drop_prompt = False
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob

        if drop_prompt:
            instruction = self.apply_chat_template("", self.SYSTEM_PROMPT_DROP)
        else:
            prompt = random.choice(self.get_segment_prompt_list)
            instruction = self.apply_chat_template(prompt, self.SYSTEM_PROMPT)

        # --- Resolve the input-image path --------------------------------
        # ``data_item['text']`` is a single line from the SAM txt list. We
        # support both absolute paths (as in BAGEL's sam_190k.py) and paths
        # relative to ``OMNIGEN2_SAM_ROOT`` so users can keep the list
        # portable across machines.
        raw_path = data_item['text']
        if os.path.isabs(raw_path):
            input_image_path = raw_path
        else:
            input_image_path = os.path.join(self.sam_base, raw_path)

        # Derive the mask path exactly the way sam_190k.py does.
        output_image_path = (
            input_image_path
            .replace('sam_selection', 'sam_mask')
            .replace('.jpg', '.png')
        )

        input_images_path = [input_image_path]
        inputs_items = None

        if not drop_ref_img:
            max_input_pixels = (
                self.max_input_pixels[0]
                if isinstance(self.max_input_pixels, list)
                else self.max_input_pixels
            )
            input_images = []
            for p in input_images_path:
                img = Image.open(p).convert("RGB")
                img = self.image_processor.preprocess(
                    img,
                    max_pixels=max_input_pixels,
                    max_side_length=self.max_side_length,
                )
                input_images.append(img)

            # Build MLLM-ready tokens + pixel features, matching
            # process_panoptic / process_semantic.
            original_padding_side = self.processor.tokenizer.padding_side
            self.processor.tokenizer.padding_side = "left"
            try:
                inputs_items = self.processor(
                    text=instruction,
                    images=input_images_path,
                    padding=True,
                    return_tensors="pt",
                )
            finally:
                self.processor.tokenizer.padding_side = original_padding_side
        else:
            input_images_path, input_images = None, None

        # --- Target mask -------------------------------------------------
        output_image = Image.open(output_image_path).convert("RGB")
        # SAM masks may be stored at a different resolution than the image
        # (sam_190k.py resizes the mask to the image size). Delegate that
        # to image_processor.preprocess which handles target-size alignment.
        output_image = self.image_processor.preprocess(
            output_image,
            max_pixels=self.max_output_pixels,
            max_side_length=self.max_side_length,
        )

        data = {
            'task_type': data_item['task_type'],
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': output_image,
            'output_image_path': output_image_path,
        }
        if inputs_items is not None:
            data.update({
                'input_ids': inputs_items.input_ids,
                'attention_mask': inputs_items.attention_mask,
                'pixel_values': inputs_items['pixel_values'],
                'image_grid_thw': inputs_items['image_grid_thw'],
            })
        return data

    def process_sft(self, data_item):
        data_item['task_type'] = 'sft'

        # Extract question and answer from the conversations
        conversations = data_item['conversations']
        human_input = ""
        gpt_output = ""

        for conv in conversations:
            if conv['from'] == 'human':
                human_input = conv['value']
            elif conv['from'] == 'gpt':
                gpt_output = conv['value']

        # Check whether images are present
        input_file = data_item.get('image', None)
        has_image = input_file is not None and input_file != "" and input_file != []

        # If `image` is a list, check whether it is empty
        if isinstance(input_file, list):
            has_image = len(input_file) > 0

        # Initialize image-related variables
        input_images_path = None
        input_images = None
        pixel_values = None
        image_grid_thw = None

        if has_image:
            # ========== With images ==========
            # Handle image paths (single or multiple)
            if isinstance(input_file, str):
                input_images_path = [os.path.join(self.sft_base, input_file)]
            elif isinstance(input_file, list):
                input_images_path = [os.path.join(self.sft_base, f) for f in input_file]

            # Preprocess images for the (unused here, but required by the
            # downstream collator) VAE tensor list.
            input_images = []
            max_input_pixels = self.max_input_pixels[0] if isinstance(self.max_input_pixels, list) else self.max_input_pixels

            for input_image_path in input_images_path:
                try:
                    input_image = Image.open(input_image_path).convert("RGB")
                    input_image = self.image_processor.preprocess(
                        input_image,
                        max_pixels=max_input_pixels,
                        max_side_length=self.max_side_length
                    )
                    input_images.append(input_image)
                except Exception as e:
                    print(f"Warning: Failed to load image {input_image_path}: {e}")
                    # Fall back to a text-only mode if image loading fails
                    has_image = False
                    input_images = None
                    input_images_path = None
                    break

        # ---------- Build the SFT chat prompt --------------------------------
        # IMPORTANT: ``apply_chat_template`` used by the generation branches
        # hardcodes exactly one ``<|image_pad|>`` placeholder and is wrong
        # for SFT because (a) LLaVA samples already use ``<image>`` markers
        # in the human turn, and the number of markers equals the number of
        # attached images; (b) text-only LLaVA samples (~17% of the jsonl)
        # have zero images, so any unconditionally-inserted image placeholder
        # causes a count mismatch between ``<|image_pad|>`` tokens and the
        # batch's ``pixel_values`` / ``image_grid_thw`` -- which surfaces as
        # an ``indexSelectLargeIndex`` CUDA assert deep inside
        # ``Qwen2_5_VL.get_image_features``.
        #
        # For SFT we therefore build the prompt directly: replace every
        # ``<image>`` in the user turn with Qwen's
        # ``<|vision_start|><|image_pad|><|vision_end|>`` triple, producing
        # exactly one placeholder per real image. Text-only samples emit no
        # placeholder at all.
        n_images_in_prompt = 0
        if has_image and input_images_path is not None:
            # If the human turn already contains ``<image>``, respect its
            # count/positions; otherwise fall back to prepending one
            # placeholder per image so the placeholder count still matches
            # ``pixel_values``.
            placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
            if '<image>' in human_input:
                # Replace every ``<image>`` (and its common ``<image>\n`` variant
                # in LLaVA) with the Qwen vision placeholder.
                user_turn = human_input.replace('<image>\n', placeholder) \
                                       .replace('<image>', placeholder)
                n_images_in_prompt = user_turn.count(placeholder)
            else:
                n_images_in_prompt = len(input_images_path)
                user_turn = (placeholder * n_images_in_prompt) + human_input

            # Keep placeholder count and image count aligned. If the text
            # under-specifies (< images) prepend extras; if it over-specifies
            # (> images) drop the surplus placeholders so processor does not
            # try to consume images that do not exist.
            if n_images_in_prompt < len(input_images_path):
                missing = len(input_images_path) - n_images_in_prompt
                user_turn = (placeholder * missing) + user_turn
                n_images_in_prompt = len(input_images_path)
            elif n_images_in_prompt > len(input_images_path):
                extra = n_images_in_prompt - len(input_images_path)
                for _ in range(extra):
                    idx = user_turn.rfind(placeholder)
                    user_turn = user_turn[:idx] + user_turn[idx + len(placeholder):]
                n_images_in_prompt = len(input_images_path)
        else:
            # Text-only: strip any stray ``<image>`` tokens so they don't
            # show up literally in the prompt.
            user_turn = human_input.replace('<image>\n', '').replace('<image>', '')

        instruction = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_turn}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Temporarily switch padding side
        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"

        try:
            if has_image and input_images_path:
                # ========== Images present: process text and images together ==========
                # Feed Qwen the *PIL* images so its internal pipeline
                # (resize -> patch_embed -> grid_thw) stays self-consistent.
                qwen_images = [
                    Image.open(p).convert("RGB") for p in input_images_path
                ]
                inputs_items = self.processor(
                    text=instruction,
                    images=qwen_images,
                    padding=True,
                    return_tensors="pt",
                )
                pixel_values = inputs_items.get('pixel_values', None)
                image_grid_thw = inputs_items.get('image_grid_thw', None)
            else:
                # ========== No images: process text only ==========
                # ``instruction`` has already been stripped of ``<image>``
                # and never inserts a ``<|image_pad|>`` when ``has_image``
                # is False, so we can tokenize it directly.
                inputs_items = self.processor.tokenizer(
                    instruction,
                    padding=True,
                    return_tensors="pt",
                )
        finally:
            # Restore padding side
            self.processor.tokenizer.padding_side = original_padding_side
        
        # Handle labels (the GPT-response portion)
        labels_tokens = self.processor.tokenizer(
            gpt_output,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        
        data = {
            'task_type': data_item['task_type'],
            'id': data_item.get('id', ''),
            'category': data_item.get('category', ''),
            'dataset': data_item.get('dataset', ''),
            'instruction': instruction,
            'input_images_path': input_images_path,
            'input_images': input_images,
            'output_image': None,
            'output_image_path': None,
            'answer': gpt_output,
            'input_ids': inputs_items.input_ids if hasattr(inputs_items, 'input_ids') else inputs_items['input_ids'],
            'attention_mask': inputs_items.attention_mask if hasattr(inputs_items, 'attention_mask') else inputs_items['attention_mask'],
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'labels': labels_tokens.input_ids,
        }
        
        return data

    def __getitem__(self, index):
        max_retries = 10
        max_total_length = 3072
        
        for attempt in range(max_retries):
            current_index = (index + attempt) % len(self.data)
            
            try:
                data_item = copy.deepcopy(self.data[current_index])#
                # pdb.set_trace()
                # Route SAM samples (txt entries pointing at sam_selection/*)
                # to process_sam; everything else falls back to the original
                # panoptic / sft heuristics below.
                if 'text' in data_item and isinstance(data_item.get('text'), str) \
                        and 'sam_selection' in data_item['text']:
                    result = self.process_sam(data_item)
                elif 'id' not in data_item or data_item['id'] is None:
                    result = self.process_panoptic(data_item)
                elif data_item['id'] is not None:
                    result = self.process_sft(data_item)
                else:
                    continue
                    # result = self.process_item(data_item)
                # # Process data...
                # result = self.process_sft(data_item)
                
                # Check length
                if result and 'input_ids' in result and 'labels' in result:
                    prompt_len = result['input_ids'].shape[-1]
                    answer_len = result['labels'].shape[-1]
                    total_len = prompt_len + answer_len
                    
                    if total_len > max_total_length:
                        print(f"[SKIP] index={current_index}, length={total_len}")
                        continue  # Too long; skip
                    else:
                        return result  # ✅ Length OK, return
                else: # gen
                    return result
                    
            except Exception as e:
                print(f"[ERROR] index={current_index}: {e}")
                continue

        return None
        
    def __len__(self):
        return len(self.data)

class OmniGen2Collator():
    def __init__(self, tokenizer, max_token_len):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __call__(self, batch):
        batch = [item for item in batch if item is not None]
    
        # If everything is None, return None (the training loop must handle it)
        if len(batch) == 0:
            return None
        def safe_get_list(key):
            """Safely fetch a list of fields from the batch"""
            result = [data.get(key) for data in batch]
            # If everything is None, return None directly
            if all(r is None for r in result):
                return None
            return result
        
        def safe_stack(tensor_list):
            """Stack a list of tensors, transparently skipping ``None`` entries.

            Originally this returned the raw list when any entry was None,
            which broke downstream ``.to(device)`` calls in train.py.
            Skipping ``None`` is the correct behaviour for Qwen-style
            ``pixel_values`` / ``image_grid_thw``: those tensors are
            *per-image*, not per-sample, so samples without an image should
            simply contribute zero rows.
            """
            if tensor_list is None:
                return None
            valid_tensors = [t for t in tensor_list if t is not None]
            if len(valid_tensors) == 0:
                return None

            # Ensure each tensor is at least 2D
            processed_tensors = []
            for t in valid_tensors:
                if t.dim() == 1:
                    t = t.unsqueeze(0)  # (3,) -> (1, 3)
                processed_tensors.append(t)

            return torch.cat(processed_tensors, dim=0)
        
        def safe_pad_sequence(tensor_list, padding_value=0):
            """Safely pad a list of tensors"""
            if tensor_list is None:
                return None
            valid_tensors = [t for t in tensor_list if t is not None]
            if len(valid_tensors) == 0:
                return None
            if len(valid_tensors) != len(tensor_list):
                return tensor_list  # Some entries are None; keep the list as-is
            tensors = [t.squeeze(0) if t.dim() > 1 else t for t in valid_tensors]
            return torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True, padding_value=padding_value
            )

        # Fetch each field
        task_type = safe_get_list('task_type')
        text_ids = safe_get_list('input_ids')
        attention_mask = safe_get_list('attention_mask')
        pixel_values = safe_get_list('pixel_values')
        image_grid_thw = safe_get_list('image_grid_thw')
        input_images = safe_get_list('input_images')
        input_images_path = safe_get_list('input_images_path')
        output_image = safe_get_list('output_image')
        output_image_path = safe_get_list('output_image_path')
        answer = safe_get_list('answer')
        labels = safe_get_list('labels')
        ids = safe_get_list('id')
        category = safe_get_list('category')
        dataset = safe_get_list('dataset')

        pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0
        # pdb.set_trace()
        data = {
            "task_type": task_type,
            "text_ids": safe_pad_sequence(text_ids, padding_value=pad_token_id),
            "text_mask": safe_pad_sequence(attention_mask, padding_value=0),
            "input_images": input_images, 
            "input_images_path": input_images_path,
            "output_image": output_image,
            "output_image_path": output_image_path,
            "pixel_values": safe_stack(pixel_values),
            "image_grid_thw": safe_stack(image_grid_thw),
            "answer": answer,
            "labels": safe_pad_sequence(labels, padding_value=-100),
            "ids": ids,
            "category": category,
            "dataset": dataset,
        }
        # pdb.set_trace()
        return data