# omnigen2/dataset/task_type_sampler.py

import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from typing import Iterator, List, Optional, Sequence


class TaskTypeDistributedSampler(Sampler[int]):
    """Distributed sampler with two operating modes.

    Legacy (``interleave_pattern`` is None, default):
        Build full global batches of a single task type so every sample in
        a global batch has the same task. This is the original behaviour.

    Interleave (``interleave_pattern`` is set):
        Produce a stream of *micro-batches* that follows a fixed task-type
        pattern, for example ``['gen', 'gen', 'sft']`` ->
        ``sam sam sft sam sam sft ...``. This lets a single-GPU run mix
        tasks across gradient-accumulation steps without ever putting two
        different task types inside the same micro-batch (which would
        trigger the ``is_mixed_batch`` path in ``train.py``).

        Within each task type the indices are drawn from the same shuffled
        permutation used by the legacy path, so sample coverage is still
        epoch-balanced per task.
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
        interleave_pattern: Optional[Sequence[str]] = None,
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
        # Normalise the pattern: canonicalise each entry to 'sft' / 'gen'.
        if interleave_pattern is not None:
            normed: List[str] = []
            for p in interleave_pattern:
                key = str(p).lower()
                if key in ('sft', 'text', 'vqa'):
                    normed.append('sft')
                else:
                    # panoptic / sam / edit / ... -> generation
                    normed.append('gen')
            if len(normed) == 0:
                raise ValueError("interleave_pattern must be non-empty")
            self.interleave_pattern = normed
        else:
            self.interleave_pattern = None

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
        if self.interleave_pattern is None:
            num_sft_batches = len(self.sft_indices) // self.global_batch_size
            num_gen_batches = len(self.gen_indices) // self.global_batch_size

            total_batches = num_sft_batches + num_gen_batches
            self.num_samples = total_batches * self.batch_size
            self.total_size = total_batches * self.global_batch_size
            return

        # Interleave mode: figure out how many full pattern cycles we can
        # run before exhausting whichever task pool runs out first.
        sft_per_cycle = self.interleave_pattern.count('sft')
        gen_per_cycle = self.interleave_pattern.count('gen')

        num_sft_batches = len(self.sft_indices) // self.global_batch_size
        num_gen_batches = len(self.gen_indices) // self.global_batch_size

        cycles_sft = num_sft_batches // sft_per_cycle if sft_per_cycle > 0 else float('inf')
        cycles_gen = num_gen_batches // gen_per_cycle if gen_per_cycle > 0 else float('inf')
        num_cycles = int(min(cycles_sft, cycles_gen))

        total_batches = num_cycles * len(self.interleave_pattern)
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

        # Chunk each pool into global batches.
        def _chunk(lst):
            return [
                lst[i:i + self.global_batch_size]
                for i in range(0, len(lst), self.global_batch_size)
                if len(lst[i:i + self.global_batch_size]) == self.global_batch_size
            ]

        sft_batches = _chunk(sft_indices_shuffled)
        gen_batches = _chunk(gen_indices_shuffled)

        if self.interleave_pattern is None:
            # Legacy: all SFT first, all GEN second, then shuffle.
            all_batches = sft_batches + gen_batches
            if self.shuffle:
                batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
                all_batches = [all_batches[i] for i in batch_perm]
        else:
            # Interleave: consume ``sft_batches`` / ``gen_batches`` following
            # ``self.interleave_pattern``. Each step of the pattern pops one
            # global batch from the corresponding pool. We stop as soon as
            # either pool cannot supply the next requested batch, so the
            # epoch ends cleanly on a pattern boundary.
            all_batches = []
            sft_it = iter(sft_batches)
            gen_it = iter(gen_batches)
            while True:
                cycle = []
                ok = True
                for task in self.interleave_pattern:
                    src = sft_it if task == 'sft' else gen_it
                    nb = next(src, None)
                    if nb is None:
                        ok = False
                        break
                    cycle.append(nb)
                if not ok:
                    break
                all_batches.extend(cycle)

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