# omnigen2/dataset/task_type_sampler.py

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Iterator, Optional

class TaskTypeDistributedSampler(Sampler[int]):
    """
    Distributed sampler that keeps every sample in a global batch from the same task type
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Build the task-type index
        if hasattr(dataset, 'sft_indices') and hasattr(dataset, 'gen_indices'):
            self.sft_indices = dataset.sft_indices.copy()
            self.gen_indices = dataset.gen_indices.copy()
        else:
            # Build the index manually if the dataset does not provide one
            self._build_indices()
        
        self.global_batch_size = batch_size * num_replicas
        self._compute_num_samples()
    
    def _build_indices(self):
        """Manually build the task-type index"""
        self.sft_indices = []
        self.gen_indices = []
        
        for i in range(len(self.dataset)):
            task_type = self.dataset.get_task_type(i) if hasattr(self.dataset, 'get_task_type') else 'gen'
            if task_type == 'sft':
                self.sft_indices.append(i)
            else:
                self.gen_indices.append(i)
    
    def _compute_num_samples(self):
        num_sft_batches = len(self.sft_indices) // self.global_batch_size
        num_gen_batches = len(self.gen_indices) // self.global_batch_size
        
        total_batches = num_sft_batches + num_gen_batches
        self.num_samples = total_batches * self.batch_size
        self.total_size = total_batches * self.global_batch_size
    
    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        if self.shuffle:
            sft_perm = torch.randperm(len(self.sft_indices), generator=g).tolist()
            gen_perm = torch.randperm(len(self.gen_indices), generator=g).tolist()
        else:
            sft_perm = list(range(len(self.sft_indices)))
            gen_perm = list(range(len(self.gen_indices)))
        
        sft_indices_shuffled = [self.sft_indices[i] for i in sft_perm]
        gen_indices_shuffled = [self.gen_indices[i] for i in gen_perm]
        
        all_batches = []
        
        for i in range(0, len(sft_indices_shuffled), self.global_batch_size):
            batch = sft_indices_shuffled[i:i + self.global_batch_size]
            if len(batch) == self.global_batch_size:
                all_batches.append(batch)
        
        for i in range(0, len(gen_indices_shuffled), self.global_batch_size):
            batch = gen_indices_shuffled[i:i + self.global_batch_size]
            if len(batch) == self.global_batch_size:
                all_batches.append(batch)
        
        if self.shuffle:
            batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_perm]
        
        indices = []
        for batch in all_batches:
            start = self.rank * self.batch_size
            end = start + self.batch_size
            indices.extend(batch[start:end])
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch