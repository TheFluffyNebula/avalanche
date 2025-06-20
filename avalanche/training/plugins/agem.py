import warnings
import random
from typing import Any, Iterator, List, Optional
import torch
from torch import Tensor

from avalanche.benchmarks.utils import _make_taskaware_classification_dataset
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.benchmarks.utils.data_loader import (
    GroupBalancedInfiniteDataLoader,
)
from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class AGEMPlugin(SupervisedPlugin):
    """Average Gradient Episodic Memory Plugin.

    AGEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. If the dot product
    between the current gradient and the (average) gradient of a randomly
    sampled set of memory examples is negative, the gradient is projected.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_experience: int, sample_size: int):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        """

        super().__init__()
        
        '''
        "Patterns" refer to the stored samples from an experience (task). 
        When a new experience arrives, we randomly sample up to `patterns_per_experience` samples from it and store in RAM
        '''
        self.patterns_per_experience = int(patterns_per_experience)

        '''
        "Samples", on the other hand, refers to the reference gradient. A-GEM randomly (while ensuring no single task dominates)
        obtains a batch of samples from the union of ALL tasks memory buffers. These are used to calculate g_ref for the constraint.
        '''
        self.sample_size = int(sample_size)
        
        # One AvalancheDataset for each experience
        self.buffers: List[AvalancheDataset] = []
        '''
        This dataloader draws minibatches of size (sample_size // number_of_buffers) from each buffer.
        '''
        self.buffer_dataloader: Optional[GroupBalancedInfiniteDataLoader] = None
        # Placeholder iterator to avoid typing issues
        self.buffer_dliter: Iterator[Any] = iter([])
        '''
        This is g_ref
        '''
        # Placeholder Tensor to avoid typing issues
        self.reference_gradients: Tensor = torch.empty(0)

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute reference gradient on memory sample.
        """

        if len(self.buffers) > 0:
            strategy.model.train()
            strategy.optimizer.zero_grad()
            mb = self.sample_from_memory()
            xref, yref, tid = mb[0], mb[1], mb[-1]
            xref, yref = xref.to(strategy.device), yref.to(strategy.device)

            out = avalanche_forward(strategy.model, xref, tid)
            loss = strategy._criterion(out, yref)
            loss.backward()
            # gradient can be None for some head on multi-headed models
            reference_gradients_list = [
                (
                    p.grad.view(-1)
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=strategy.device)
                )
                for n, p in strategy.model.named_parameters()
            ]
            self.reference_gradients = torch.cat(reference_gradients_list)
            strategy.optimizer.zero_grad()

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        if len(self.buffers) > 0:
            current_gradients_list = [
                (
                    p.grad.view(-1)
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=strategy.device)
                )
                for n, p in strategy.model.named_parameters()
            ]
            current_gradients = torch.cat(current_gradients_list)

            assert (
                current_gradients.shape == self.reference_gradients.shape
            ), "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(
                    self.reference_gradients, self.reference_gradients
                )
                grad_proj = current_gradients - self.reference_gradients * alpha2

                count = 0
                for n, p in strategy.model.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(grad_proj[count : count + n_param].view_as(p))
                    count += n_param
    
    '''
    This method is probably defined by the 'SupervisedPlugin' interface. This method is just a wrapper for update_memory.
    '''
    def after_training_exp(self, strategy, **kwargs):
        """Update replay memory with patterns from current experience."""
        self.update_memory(strategy.experience.dataset, **kwargs)

    def sample_from_memory(self):
        """
        Sample a minibatch from memory.
        Return a tuple of patterns (tensor), targets (tensor).
        """
        return next(self.buffer_dliter)

    @torch.no_grad()
    def update_memory(self, dataset, num_workers=0, **kwargs):
        """
        Update replay memory with patterns from current experience.
        """
        if num_workers > 0:
            warnings.warn(
                "Num workers > 0 is known to cause heavy" "slowdowns in AGEM."
            )

        '''
        This section ensures that if a experience (task) has more than `patterns_per_experience` samples, 
        we randomly shuffle its indices and keep only that many. This mimics a uniform random sampling approach,
        as defined in the A-GEM paper.
        '''
        removed_els = len(dataset) - self.patterns_per_experience
        if removed_els > 0:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            dataset = dataset.subset(indices[: self.patterns_per_experience])

        self.buffers.append(dataset)
        '''
        This part updates the dataloader to include the newly completed task in its minibatch sampling algorithm.
        Recall that this dataloader ensures even distribution of samples across all k < t tasks when 
        calculating g_ref.
        '''
        persistent_workers = num_workers > 0
        self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
            self.buffers,
            batch_size=(self.sample_size // len(self.buffers)),
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=persistent_workers,
        )
        '''
        We gotta update the iterator to match the newly updated dataloader.
        '''
        self.buffer_dliter = iter(self.buffer_dataloader)
