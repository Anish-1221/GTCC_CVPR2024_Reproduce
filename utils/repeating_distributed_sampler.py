"""
Repeating Distributed Sampler for small datasets in DDP training.

This sampler extends PyTorch's DistributedSampler behavior to ensure that tasks with
fewer samples than (batch_size * world_size) still contribute to training
by repeating samples.

Problem: When dataset_size < batch_size * world_size, with drop_last=True in DataLoader,
zero batches are formed and the task is completely skipped during training.

Solution: Repeat dataset indices enough times to guarantee at least one batch per GPU.
"""

import math
from typing import Optional, Iterator, TypeVar
import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

T_co = TypeVar('T_co', covariant=True)


class RepeatingDistributedSampler(Sampler[int]):
    """
    A distributed sampler that repeats dataset indices to ensure at least
    one complete batch can be formed per GPU when using drop_last=True.

    This solves the problem where small datasets (N < batch_size * world_size)
    would yield zero batches in standard DistributedSampler + drop_last setup.

    Args:
        dataset: Dataset to sample from
        batch_size: Batch size per GPU (used to calculate minimum samples needed)
        num_replicas: Number of distributed processes (GPUs). Defaults to world_size.
        rank: Rank of current process. Defaults to current rank.
        shuffle: Whether to shuffle indices each epoch
        seed: Random seed for shuffling
        drop_last: Whether DataLoader will use drop_last (affects repeat calculation)

    Example:
        If dataset has 6 samples, batch_size=4, world_size=8:
        - Standard DistributedSampler: each GPU gets 1 sample, 0 batches with drop_last
        - RepeatingDistributedSampler: repeats to 48 samples, each GPU gets 6 samples,
          allowing at least 1 batch of 4 per GPU

    Usage:
        sampler = RepeatingDistributedSampler(
            dataset=train_set,
            batch_size=4,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
            seed=42,
            drop_last=True
        )
        dataloader = DataLoader(
            train_set,
            batch_size=4,
            sampler=sampler,
            drop_last=True
        )
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        # Handle distributed setup
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

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, should be in range [0, {num_replicas})"
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Calculate how many times to repeat the dataset
        self.dataset_size = len(dataset)
        self.repeat_factor = self._calculate_repeat_factor()
        self.effective_size = self.dataset_size * self.repeat_factor

        # Calculate samples per GPU (with padding for even distribution)
        self.num_samples = math.ceil(self.effective_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def _calculate_repeat_factor(self) -> int:
        """
        Calculate how many times to repeat the dataset to ensure at least
        batch_size samples per GPU.

        For drop_last=True, we need each GPU to have at least batch_size samples.
        Total samples needed = batch_size * num_replicas

        Returns:
            int: Number of times to repeat the dataset (minimum 1)
        """
        if not self.drop_last:
            # Without drop_last, we don't need to repeat
            return 1

        if self.dataset_size == 0:
            return 1

        # Total samples needed across all GPUs for at least 1 batch each
        # Each GPU needs at least batch_size samples
        total_needed = self.batch_size * self.num_replicas

        if self.dataset_size >= total_needed:
            return 1

        # Calculate repeat factor to get enough samples
        repeat_factor = math.ceil(total_needed / self.dataset_size)

        return repeat_factor

    def __iter__(self) -> Iterator[int]:
        # Create repeated indices with shuffling
        if self.shuffle:
            # Different shuffle each epoch, but deterministic given seed + epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # For repeated dataset, we shuffle each repeat differently
            # then interleave them to avoid clustering of identical samples
            repeated_indices = []
            for rep in range(self.repeat_factor):
                # Each repeat gets a different shuffle based on epoch and rep number
                rep_g = torch.Generator()
                rep_g.manual_seed(self.seed + self.epoch * 1000 + rep)
                perm = torch.randperm(self.dataset_size, generator=rep_g).tolist()
                repeated_indices.extend(perm)

            # Final shuffle of the full repeated list to interleave samples
            indices = [repeated_indices[i] for i in
                      torch.randperm(len(repeated_indices), generator=g).tolist()]
        else:
            # No shuffle: just tile the indices
            indices = list(range(self.dataset_size)) * self.repeat_factor

        # Pad if needed to make evenly divisible by num_replicas
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            # Pad by repeating from the beginning
            indices += indices[:padding_size]

        assert len(indices) == self.total_size, \
            f"Expected {self.total_size} indices, got {len(indices)}"

        # Subsample for this rank (interleaved distribution)
        # rank 0 gets indices 0, num_replicas, 2*num_replicas, ...
        # rank 1 gets indices 1, num_replicas+1, 2*num_replicas+1, ...
        indices = indices[self.rank:self.total_size:self.num_replicas]

        assert len(indices) == self.num_samples, \
            f"Expected {self.num_samples} samples for rank {self.rank}, got {len(indices)}"

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.

        This ensures different shuffling across epochs while maintaining
        reproducibility. Must be called before each epoch in the training loop.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch

    def get_repeat_info(self) -> dict:
        """
        Return diagnostic information about the sampler configuration.

        Useful for logging and debugging.

        Returns:
            dict: Information about repeat factor, effective size, etc.
        """
        return {
            'original_dataset_size': self.dataset_size,
            'repeat_factor': self.repeat_factor,
            'effective_dataset_size': self.effective_size,
            'samples_per_gpu': self.num_samples,
            'total_size_with_padding': self.total_size,
            'batch_size': self.batch_size,
            'world_size': self.num_replicas,
            'needs_repetition': self.repeat_factor > 1,
            'batches_per_gpu': self.num_samples // self.batch_size if self.drop_last else math.ceil(self.num_samples / self.batch_size),
        }
