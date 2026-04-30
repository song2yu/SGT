import dotenv
import pdb
dotenv.load_dotenv(override=True)
import os
# Set WANDB_API_KEY via environment variable or a .env file; do not hardcode it here.
os.environ.setdefault("WANDB_MODE", "offline")
import time
from copy import deepcopy
import argparse
import logging
import math
import os
import shutil
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

import torch

import torch.nn.functional as F
import torch.utils.checkpoint

from torchvision.transforms.functional import crop, to_pil_image, to_tensor

from einops import repeat, rearrange

import accelerate
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import transformers
from transformers import AutoTokenizer
# from transformers import Qwen2_5_VLModel as TextEncoder
from transformers import Qwen2_5_VLForConditionalGeneration as TextEncoder


import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from peft import LoraConfig

from omnigen2.training_utils import EMAModel
from omnigen2.utils.logging_utils import TqdmToLogger
from omnigen2.transport import create_transport
from omnigen2.dataset.omnigen2_train_dataset import OmniGen2TrainDataset, OmniGen2Collator
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed


logger = get_logger(__name__)

    
def parse_args(root_path) -> OmegaConf:
    parser = argparse.ArgumentParser(description="OmniGen2 training script")
    parser.add_argument(
        "--config",
        type=str,
        default='options/ft_lora.yml',
        help="Path to configuration file (YAML format)",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        help="Global batch size.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        # default=None,  # e.g., 'data_configs/train/example/edit/edit.yml'
        help="Data path.",
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    # Output directory is resolved relative to the project root so the code
    # is portable across machines. Override `root_dir` via config if needed.
    output_dir = os.path.join(root_path, 'experiments', conf.name)
    conf.root_dir = root_path
    conf.output_dir = output_dir
    conf.config_file = args.config

    # Override config with command line arguments
    if args.global_batch_size is not None:
        conf.train.global_batch_size = args.global_batch_size
    
    if args.data_path is not None:
        conf.data.data_path = args.data_path
    return conf

def setup_logging(args: OmegaConf, accelerator: Accelerator) -> None:
    """
    Set up logging configuration for training.
    
    Args:
        accelerator: Accelerator instance
        args: Configuration object
        logging_dir: Directory for log files
    """

    logging_dir = Path(args.output_dir, "logs")
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config_file, args.output_dir)
        
        # Create logging directory and file handler
        os.makedirs(logging_dir, exist_ok=True)
        log_file = Path(logging_dir, f'{time.strftime("%Y%m%d-%H%M%S")}.log')

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.logger.addHandler(file_handler)

    # Configure basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set verbosity for different processes
    log_level = logging.INFO if accelerator.is_local_main_process else logging.ERROR
    transformers.utils.logging.set_verbosity(log_level)
    diffusers.utils.logging.set_verbosity(log_level)


def log_model_info(name: str, model: torch.nn.Module):
    """Logs parameter counts for a given model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"--- {name} ---")
    logger.info(model)
    logger.info(f"Total parameters (M): {total_params / 1e6:.2f}")
    logger.info(f"Trainable parameters (M): {trainable_params / 1e6:.2f}")


def log_time_distribution(transport, device, args):
    """Samples time steps from transport and plots their distribution."""
    with torch.no_grad():
        dummy_tensor = torch.randn((64, 16, int(math.sqrt(args.data.max_output_pixels) / 8), int(math.sqrt(args.data.max_output_pixels) / 8)), device=device)
        ts = torch.cat([transport.sample(dummy_tensor, AcceleratorState().process_index, AcceleratorState().num_processes)[0] for _ in range(1000)], dim=0)
    
    ts_np = ts.cpu().numpy()
    percentile_70 = np.percentile(ts_np, 70)
    
    plt.figure(figsize=(10, 6))
    plt.hist(ts_np, bins=50, edgecolor='black', alpha=0.7, label="Time Step Distribution")
    plt.axvline(percentile_70, color='red', linestyle='dashed', linewidth=2, label=f'70th Percentile = {percentile_70:.2f}')
    plt.title('Distribution of Sampled Time Steps (t)')
    plt.xlabel('Time Step (t)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = Path(args.output_dir) / 't_distribution.png'
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Time step distribution plot saved to {save_path}")
    
    
class OmniGenJointModel(torch.nn.Module):
    def __init__(self, transformer, text_encoder):
        super().__init__()
        self.model = transformer          # DiT model
        self.text_encoder = text_encoder  # Qwen2.5-VL
        
        # Retrieve the LM head (used for SFT).
        # For Qwen2.5-VL it is usually available as `model.lm_head`.
        if hasattr(text_encoder, 'lm_head'):
            self.lm_head = text_encoder.lm_head
        else:
            # Fall back to creating one manually if unavailable.
            hidden_size = text_encoder.config.hidden_size
            vocab_size = text_encoder.config.vocab_size
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            print(f"Created new lm_head: hidden_size={hidden_size}, vocab_size={vocab_size}")
    
    def forward(self, *args, **kwargs):
        """
        Unified forward entry point.
        """
        # Mode 1: Text encoder only (return features, no loss).
        if kwargs.pop("run_text_encoder", False):
            kwargs.pop("labels", None)
            return self._forward_text_encoder(*args, **kwargs)
        
        # Mode 2: SFT (run text encoder and compute loss).
        if kwargs.pop("run_sft", False):
            return self._forward_sft(*args, **kwargs)
        
        # Mode 3: DiT / Transformer (default).
        return self._forward_dit(*args, **kwargs)
    
    def _forward_text_encoder(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        output_hidden_states=False,
        **kwargs
    ):
        """Return hidden states without computing the loss."""
        return self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=output_hidden_states,
            **kwargs
        )
    
    def _forward_sft(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        labels=None,
        output_hidden_states=False,
        **kwargs
    ):
        # import torch
        # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # if rank == 0:
        #     print(f"\n[DEBUG] input_ids shape: {input_ids.shape}")
        #     if pixel_values is not None:
        #         print(f"[DEBUG] pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
        #     if image_grid_thw is not None:
        #         print(f"[DEBUG] image_grid_thw: {image_grid_thw}")
        #     # GPU memory usage
        #     print(f"[DEBUG] GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        """Call the ForConditionalGeneration model directly; it computes the loss internally."""
        with torch.backends.cudnn.flags(enabled=False):  # Disable cuDNN
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,  # Labels are passed through directly.
                output_hidden_states=output_hidden_states,
                **kwargs
            )
        
        # outputs.loss is already computed by the model.
        return outputs
        
    def _forward_dit(
        self,
        x=None,
        t=None,
        text_hidden_states=None,
        text_attention_mask=None,
        ref_image_hidden_states=None,
        freqs_cis=None,
        **kwargs
    ):
        """Run the DiT."""
        return self.model(
            hidden_states=x,
            timestep=t,
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            ref_image_hidden_states=ref_image_hidden_states,
            freqs_cis=freqs_cis,
            **kwargs
        )


# Custom output dataclass
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class SFTOutput:
    """Output container for the SFT task."""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple] = None


def main(args):
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=Path(args.output_dir, 'logs'))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        mixed_precision=args.train.mixed_precision,
        log_with=OmegaConf.to_object(args.logger.log_with),
        project_config=accelerator_project_config,
    )

    setup_logging(args, accelerator)
    
    # Reproducibility
    if args.seed is not None:
        set_seed(args.seed, device_specific=args.get('device_specific_seed', False))

    # Set performance flags
    if args.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.train.get('benchmark_cudnn', False):
        torch.backends.cudnn.benchmark = True

    weight_dtype = torch.bfloat16
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    ema_decay = args.train.get('ema_decay', 0)

    model = OmniGen2Transformer2DModel.from_pretrained(
        args.model.pretrained_model_path, subfolder="transformer", torch_dtype=weight_dtype#,device_map='auto'
    )
    model.to(dtype=weight_dtype)
    model.train()

    # model = OmniGen2Transformer2DModel(**args.model.arch_opt)
    # model.train()

    freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
        model.config.axes_dim_rope,
        model.config.axes_lens,
        theta=10000,
    )

    # if args.model.get("pretrained_model_path", None) is not None:
    #     logger.info(f"Loading model parameters from: {args.model.pretrained_model_path}")
    #     state_dict = torch.load(args.model.pretrained_model_path, map_location="cpu")
    #     missing, unexpect = model.load_state_dict(state_dict, strict=False)
    #     logger.info(
    #         f"missed parameters: {missing}",
    #     )
    #     logger.info(f"unexpected parameters: {unexpect}")
    
    if ema_decay != 0:
        model_ema = deepcopy(model)
        model_ema._requires_grad = False

    text_tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained_text_encoder_model_name_or_path)
    text_tokenizer.padding_side = "right"

    if accelerator.is_main_process:
        text_tokenizer.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))

    text_encoder = TextEncoder.from_pretrained(
        args.model.pretrained_text_encoder_model_name_or_path,
        torch_dtype=weight_dtype,
    )
    if args.model.get('resize_token_embeddings', False):
        text_encoder.resize_token_embeddings(len(text_tokenizer))

    if accelerator.is_main_process:
        text_encoder.save_pretrained(os.path.join(args.output_dir, 'text_encoder'))

    # log_model_info("text_encoder", text_encoder)

    vae = AutoencoderKL.from_pretrained(
        args.model.pretrained_vae_model_name_or_path,
        subfolder=args.model.get("vae_subfolder", "vae"),
    )
    
    logger.info(vae)
    logger.info("***** Move vae, text_encoder to device and cast to weight_dtype *****")
    # Move vae, unet, text_encoder and controlnet_ema to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    args.train.lora_ft = args.train.get('lora_ft', False)
    if args.train.lora_ft:
        model.requires_grad_(False)

        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        # now we will add new LoRA weights the transformer layers
        lora_config = LoraConfig(
            r=args.train.lora_rank,
            lora_alpha=args.train.lora_rank,
            lora_dropout=args.train.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model.add_adapter(lora_config)

    if args.train.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        text_encoder.gradient_checkpointing_enable()

    model.requires_grad_(True)
    # pdb.set_trace()
    text_encoder.requires_grad_(True)
    joint_model = OmniGenJointModel(model, text_encoder)

    log_model_info("joint_model", joint_model)

    if args.train.scale_lr:
        args.train.learning_rate = (
            args.train.learning_rate * args.train.gradient_accumulation_steps * args.train.batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # log_model_info("transformer", model)

    # Optimizer creation
    # trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # train mllm: qwen 
    # trainable_params += list(filter(lambda p: p.requires_grad, text_encoder.parameters()))
    trainable_params = list(filter(lambda p: p.requires_grad, joint_model.parameters()))
    for name, param in joint_model.named_parameters():
        if param.requires_grad:
            print(f"{name:<60} {str(list(param.shape)):<20}")
    # pdb.set_trace()
    optimizer = optimizer_class(
        trainable_params,
        lr=args.train.learning_rate,
        betas=(args.train.adam_beta1, args.train.adam_beta2),
        weight_decay=args.train.adam_weight_decay,
        eps=args.train.adam_epsilon,
    )

    logger.info("***** Prepare dataset *****")

    with accelerator.main_process_first():
        train_dataset = OmniGen2TrainDataset(
            args.data.data_path,
            tokenizer=text_tokenizer,
            use_chat_template=args.data.use_chat_template,
            prompt_dropout_prob=args.data.get('prompt_dropout_prob', 0.0),
            ref_img_dropout_prob=args.data.get('ref_img_dropout_prob', 0.0),
            max_input_pixels=OmegaConf.to_object(args.data.get('max_input_pixels', 1024 * 1024)),
            max_output_pixels=args.data.get('max_output_pixels', 1024 * 1024),
            max_side_length=args.data.get('max_side_length', 2048),
        )

    # default: 1000 steps, linear noise schedule
    transport = create_transport(
        "Linear",
        "velocity",
        None,
        None,
        None,
        snr_type=args.transport.snr_type,
        do_shift=args.transport.do_shift,
        seq_len=args.data.max_output_pixels // 16 // 16,
        dynamic_time_shift=args.transport.get("dynamic_time_shift", False),
        time_shift_version=args.transport.get("time_shift_version", "v1"),
    )  # default: velocity;

    # Log time distribution for analysis
    if accelerator.is_main_process:
        log_time_distribution(transport, accelerator.device, args)

    logger.info(f"Number of training samples: {len(train_dataset)}")

    if args.seed is not None and args.get("workder_specific_seed", False):
        from omnigen2.utils.reproducibility import worker_init_fn

        worker_init_fn = partial(
            worker_init_fn,
            num_processes=AcceleratorState().num_processes,
            num_workers=args.train.dataloader_num_workers,
            process_index=AcceleratorState().process_index,
            seed=args.seed,
            same_seed_per_epoch=args.get("same_seed_per_epoch", False),
        )
    else:
        worker_init_fn = None

    logger.info("***** Prepare dataLoader *****")
    from omnigen2.dataset.sampler import TaskTypeDistributedSampler
    task_sampler = TaskTypeDistributedSampler(
        dataset=train_dataset,
        batch_size=args.train.batch_size,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=args.seed if args.seed is not None else 0,
        drop_last=True,
    )


    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # shuffle=True,
        sampler=task_sampler,
        batch_size=args.train.batch_size,
        num_workers=args.train.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        collate_fn=OmniGen2Collator(tokenizer=text_tokenizer, max_token_len=args.data.maximum_text_tokens)
    )

    logger.info(f"{args.train.batch_size=} {args.train.gradient_accumulation_steps=} {accelerator.num_processes=} {args.train.global_batch_size=}")
    assert (
        args.train.batch_size
        * args.train.gradient_accumulation_steps
        * accelerator.num_processes
        == args.train.global_batch_size
    ), (
        f"{args.train.batch_size=} * {args.train.gradient_accumulation_steps=} * {accelerator.num_processes=} should be equal to {args.train.global_batch_size=}"
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if 'max_train_steps' not in args.train:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.train.lr_scheduler == 'timm_cosine':
        from omnigen2.optim.scheduler.cosine_lr import CosineLRScheduler

        lr_scheduler = CosineLRScheduler(optimizer=optimizer,
                                         t_initial=args.train.t_initial,
                                         lr_min=args.train.lr_min,
                                         cycle_decay=args.train.cycle_decay,
                                         warmup_t=args.train.warmup_t,
                                         warmup_lr_init=args.train.warmup_lr_init,
                                         warmup_prefix=args.train.warmup_prefix,
                                         t_in_epochs=args.train.t_in_epochs)
    elif args.train.lr_scheduler == 'timm_constant_with_warmup':
        from omnigen2.optim.scheduler.step_lr import StepLRScheduler

        lr_scheduler = StepLRScheduler(
            optimizer=optimizer,
            decay_t=1,
            decay_rate=1,
            warmup_t=args.train.warmup_t,
            warmup_lr_init=args.train.warmup_lr_init,
            warmup_prefix=args.train.warmup_prefix,
            t_in_epochs=args.train.t_in_epochs,
        )
    else:
        lr_scheduler = get_scheduler(
            args.train.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.train.lr_warmup_steps,
            num_training_steps=args.train.max_train_steps,
            num_cycles=args.train.lr_num_cycles,
            power=args.train.lr_power,
        )

    logger.info("***** Prepare everything with our accelerator *****")
    if args.train.ema_decay != 0:
        joint_model, model_ema, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            joint_model, model_ema, optimizer, train_dataloader, lr_scheduler
        )
        model_ema = EMAModel(model_ema.parameters(), decay=ema_decay, model_cls=type(unwrap_model(model)), model_config=model_ema.config)
    else:
        joint_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            joint_model, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.train.num_train_epochs = math.ceil(args.train.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("OmniGen2", init_kwargs={"wandb": {"name": args.name}})

    # Train!
    total_batch_size = args.train.batch_size * accelerator.num_processes * args.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.train.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.train.max_train_steps}")
    global_step = 0
    first_epoch = 0
        
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.train.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        file=TqdmToLogger(logger, level=logging.INFO)
    )

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                logger.info(f"***** Wandb log dir: {tracker.run.dir} *****")
    
    for epoch in range(first_epoch, args.train.num_train_epochs):
        if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
            break
        task_sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            if batch is None:
                print(f"Warning: Skipping None batch at epoch {epoch}, step {step}")
                continue
            # ============ Fetch batch data ============
            task_types = batch['task_type']  # List[str]
            input_images = batch['input_images']
            output_image = batch['output_image']
            
            pixel_values = batch['pixel_values']
            image_grid_thw = batch['image_grid_thw']
            
            # ============ Determine task type ============
            batch_size = len(task_types)
            is_sft_batch = all(t == 'sft' for t in task_types)
            is_gen_batch = all(t != 'sft' for t in task_types)
            is_mixed_batch = not is_sft_batch and not is_gen_batch
            raw_prompt_ids = batch['text_ids'].to(accelerator.device)

            # ============ Prepare inputs according to task type ============
            if is_sft_batch:
                # SFT: concatenate prompt + response.
                prompt_ids = batch['text_ids'].to(accelerator.device)      # [B, prompt_len]
                response_ids = batch['labels'].to(accelerator.device)      # [B, response_len]
                prompt_mask = batch['text_mask'].to(accelerator.device)
                
                # Concatenate.
                input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
                labels = torch.cat([
                    torch.full_like(prompt_ids, -100),
                    response_ids
                ], dim=-1)
                attention_mask = torch.cat([
                    prompt_mask,
                    torch.ones_like(response_ids)
                ], dim=-1)
                # max_sft_length = 1024 * 3  # Tune based on available memory; try 1024 / 2048 / 4096.
                # if input_ids.shape[1] > max_sft_length:
                #     logger.warning(f"SFT sequence too long ({input_ids.shape[1]}), skipping batch")
                #     continue
            else:
                # Image generation task: no concatenation needed.
                input_ids = batch['text_ids'].to(accelerator.device)
                attention_mask = batch['text_mask'].to(accelerator.device)
                labels = None
            
            # ============ Handle visual inputs ============
            if pixel_values is not None:
                pixel_values = pixel_values.to(accelerator.device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(accelerator.device)
                if image_grid_thw.dim() == 1:
                    image_grid_thw = image_grid_thw.unsqueeze(0)
            
            # ============ Initialize loss bins ============
            n_loss_bins = 10
            loss_bins = torch.linspace(0.0, 1.0, n_loss_bins + 1, device=accelerator.device)
            bin_occurrence = torch.zeros(n_loss_bins, device=accelerator.device)
            bin_sum_loss = torch.zeros(n_loss_bins, device=accelerator.device)
            
            with accelerator.accumulate(joint_model):
                
                # ============ Pure SFT task ============
                if is_sft_batch:
                    sft_output = joint_model(
                        input_ids=input_ids,           # Concatenated
                        attention_mask=attention_mask, # Concatenated
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        labels=labels,                 # Concatenated (prompt part is -100)
                        output_hidden_states=False,
                        run_sft=True,
                    )
                    
                    total_loss = sft_output.loss
                    loss_dict = None
                    
                    avg_loss = accelerator.gather(total_loss.detach().repeat(batch_size)).mean()
                    avg_total_loss = avg_loss
                
                # ============ Pure image generation task ============
                elif is_gen_batch:
                    # Obtain text features.
                    text_encoder_output = joint_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        output_hidden_states=True,
                        run_text_encoder=True
                    )
                    # pdb.set_trace()
                    text_feats = text_encoder_output.hidden_states[-1]
                    # text_feats = text_encoder_output.last_hidden_state
                    
                    @torch.no_grad()
                    def encode_vae(img):
                        z0 = vae.encode(img.to(dtype=vae.dtype)).latent_dist.sample()
                        if vae.config.shift_factor is not None:
                            z0 = z0 - vae.config.shift_factor
                        if vae.config.scaling_factor is not None:
                            z0 = z0 * vae.config.scaling_factor
                        z0 = z0.to(dtype=weight_dtype)
                        return z0
                    
                    # Encode input images.
                    input_latents = []
                    for i, img in enumerate(input_images):
                        reca_flag = True
                        if img is not None and len(img) > 0 and reca_flag is False:
                            input_latents.append([encode_vae(img_j).squeeze(0) for img_j in img])
                        else:
                            input_latents.append(None)

                    # Encode output images.
                    output_latents = [encode_vae(img).squeeze(0) for img in output_image]
                    
                    model_kwargs = dict(
                        text_hidden_states=text_feats,
                        text_attention_mask=attention_mask,
                        ref_image_hidden_states=input_latents,
                        freqs_cis=freqs_cis,
                    )

                    # Count tokens.
                    local_num_tokens_in_batch = sum(latent.numel() for latent in output_latents)
                    num_tokens_in_batch = accelerator.gather(
                        torch.tensor(local_num_tokens_in_batch, device=accelerator.device)
                    ).sum().item()
                    
                    # Flow matching loss
                    loss_dict = transport.training_losses(
                        joint_model,
                        output_latents,
                        model_kwargs,
                        process_index=AcceleratorState().process_index,
                        num_processes=AcceleratorState().num_processes,
                        reduction='sum'
                    )
                    loss = loss_dict["loss"].sum()
                    loss = (loss * accelerator.gradient_state.num_steps * accelerator.num_processes) / num_tokens_in_batch
                    total_loss = loss
                    
                    # Loss bin statistics
                    bin_indices = torch.bucketize(loss_dict["t"].cuda(), loss_bins, right=True) - 1
                    detached_loss = loss_dict["loss"].detach()
                    local_num_tokens = torch.tensor([latent.numel() for latent in output_latents], device=accelerator.device)

                    for i in range(n_loss_bins):
                        mask = bin_indices == i
                        bin_occurrence[i] += local_num_tokens[mask].sum()
                        bin_sum_loss[i] += detached_loss[mask].sum()

                    avg_loss = accelerator.gather(loss.detach().repeat(batch_size)).mean() / accelerator.gradient_state.num_steps
                    avg_total_loss = accelerator.gather(total_loss.detach().repeat(batch_size)).mean() / accelerator.gradient_state.num_steps
                
                # ============ Mixed batch ============
                else:
                    # TODO: temporarily skipped; validate single-task training first.
                    raise NotImplementedError("Mixed batch not supported yet")
                
                # ============ Backward pass ============
                accelerator.backward(total_loss)
                
                # Aggregate loss bin statistics.
                if not is_sft_batch:
                    bin_occurrence = accelerator.gather(rearrange(bin_occurrence, "b -> 1 b")).sum(dim=0)
                    bin_sum_loss = accelerator.gather(rearrange(bin_sum_loss, "b -> 1 b")).sum(dim=0)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.train.max_grad_norm)

                    optimizer.step()
                    if 'timm' in args.train.lr_scheduler:
                        lr_scheduler.step(global_step)
                    else:
                        lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.train.set_grads_to_none)

            # ============ Logging (rest remains unchanged) ============
            # ... (omitted; original behavior preserved) ...

            # ============ Logging ============
            if accelerator.sync_gradients:
                logs = {
                    "loss": avg_loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0],
                    "total_loss": avg_total_loss.detach().item(),
                    "task_type": task_types[0] if len(set(task_types)) == 1 else "mixed",
                }

                # Only record loss bins for image generation tasks.
                if not is_sft_batch:
                    for i in range(n_loss_bins):
                        if bin_occurrence[i] > 0:
                            bin_avg_loss = (bin_sum_loss[i] / bin_occurrence[i]).item()
                            logs[f"loss-bin{i+1}-{n_loss_bins}"] = bin_avg_loss

                if ema_decay != 0:
                    model_ema.step(model.parameters())
                    
                global_step += 1

                # ============ Save checkpoint ============
                if global_step % args.logger.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.logger.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            
                            if len(checkpoints) >= args.logger.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.logger.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                                        
                    accelerator.wait_for_everyone()
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
                
                # ============ Visualization (image generation task only) ============
                if accelerator.is_main_process:
                    if 'train_visualization_steps' in args.val and (global_step - 1) % args.val.train_visualization_steps == 0:
                        
                        # SFT task visualization: print generated text.
                        if is_sft_batch:
                            num_samples = min(args.val.get('num_train_visualization_samples', 3), batch_size)
                            for i in range(num_samples):
                                input_ids = raw_prompt_ids[i]
                                instruction = text_tokenizer.decode(input_ids, skip_special_tokens=False)
                                answer = batch['answer'][i] if batch.get('answer') else ""
                                
                                with open(os.path.join(args.output_dir, f"sft_sample_{global_step}_{i}.txt"), "w", encoding='utf-8') as f:
                                    f.write(f"=== Input ===\n{instruction}\n\n=== Target Answer ===\n{answer}")
                        
                        # Image generation task visualization.
                        elif loss_dict is not None:
                            num_samples = min(args.val.get('num_train_visualization_samples', 3), batch_size)
                            with torch.no_grad():
                                for i in range(num_samples):
                                    if i >= len(loss_dict['pred']):
                                        break
                                        
                                    model_pred = loss_dict['pred'][i]
                                    pred_image = loss_dict['xt'][i] + (1 - loss_dict['t'][i]) * model_pred

                                    if vae.config.scaling_factor is not None:
                                        pred_image = pred_image / vae.config.scaling_factor
                                    if vae.config.shift_factor is not None:
                                        pred_image = pred_image + vae.config.shift_factor
                                    pred_image = vae.decode(
                                        pred_image.unsqueeze(0).to(dtype=weight_dtype),
                                        return_dict=False,
                                    )[0]
                                    pred_image = pred_image.clamp(-1, 1)

                                    vis_images = [output_image[i]] + [pred_image]
                                    if input_images[i] is not None:
                                        vis_images = input_images[i] + vis_images

                                    # Concatenate images of different sizes horizontally.
                                    max_height = max(img.shape[-2] for img in vis_images)
                                    total_width = sum(img.shape[-1] for img in vis_images)
                                    canvas = torch.zeros((3, max_height, total_width), device=vis_images[0].device)
                                    
                                    current_x = 0
                                    for img in vis_images:
                                        h, w = img.shape[-2:]
                                        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
                                        current_x += w
                                    
                                    to_pil_image(canvas).save(
                                        os.path.join(args.output_dir, f"gen_visualization_{global_step}_{i}_t{loss_dict['t'][i]:.3f}.png")
                                    )
                                    
                                    input_ids = raw_prompt_ids[i]
                                    instruction = text_tokenizer.decode(input_ids, skip_special_tokens=False)

                                    with open(os.path.join(args.output_dir, f"instruction_{global_step}_{i}.txt"), "w", encoding='utf-8') as f:
                                        f.write(f"token len: {len(input_ids)}\ntext: {instruction}")

                progress_bar.set_postfix(**logs)
                progress_bar.update(1)

                accelerator.log(logs, step=global_step)

            if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
                break

    checkpoints = os.listdir(args.output_dir)
    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    if len(checkpoints) > 0 and int(checkpoints[-1].split("-")[1]) < global_step:
        if accelerator.is_main_process:
            if args.logger.checkpoints_total_limit is not None:
                if len(checkpoints) >= args.logger.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.logger.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]

                    logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)

        accelerator.wait_for_everyone()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args(root_path)
    main(args)