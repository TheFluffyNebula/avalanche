from typing import Dict
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class DualGEMPlugin(SupervisedPlugin):
    """
    Gradient Episodic Memory Plugin.
    GEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. The gradient on
    the current minibatch is projected so that the dot product with all the
    reference gradients of previous tasks remains positive.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_experience: int, memory_strength: float):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()

        self.patterns_per_experience = int(patterns_per_experience)
        '''
        "In practice, we found that adding a small constant lambda to v biased the gradient projection 
            to updates that favoured beneficial backwards transfer"
        '''
        self.memory_strength = memory_strength
        # (x, t, y) triplet: x -- feature vector, t -- task descriptor, y -- target vector
        self.memory_x: Dict[int, Tensor] = dict()
        self.memory_y: Dict[int, Tensor] = dict()
        self.memory_tid: Dict[int, Tensor] = dict()
        # initialize G, the matrix for gradients on loss on for (f(Θ), memory buffer)
        self.G: Tensor = torch.empty(0)

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute gradient constraints on previous memory samples from all
        experiences.
        """
        '''
        "If we have already trained on at least one prior experience, 
        then use the saved memory gradients from those experiences to compute constraint projections."
        '''
        if strategy.clock.train_exp_counter > 0:
            G = []
            strategy.model.train()
            # iterate over previous experiences, index = t
            for t in range(strategy.clock.train_exp_counter):
                # put the model in training mode
                strategy.model.train()
                # clear old gradients before computing a new backward pass. This is crucial to prevent gradient accumulation.
                strategy.optimizer.zero_grad()
                # load memory buffer input-output pairs to the same device (cpu/gpu) as the model
                xref = self.memory_x[t].to(strategy.device)
                yref = self.memory_y[t].to(strategy.device)
                # run a forward pass through the model on these reference inputs.
                out = avalanche_forward(strategy.model, xref, self.memory_tid[t])
                # compute loss on reference examples (compare y to y-hat)
                loss = strategy._criterion(out, yref)
                # backward pass to compute gradients of model parameters with respect to loss
                loss.backward()
                '''
                Extract and flatten all parameter gradients, concatenate them into a 1D tensor, and append to list G.
                This creates a single vector representing the gradient for experience t.
                If any parameter has no gradient (None), a zero vector of matching size is used instead (to preserve shape alignment).
                '''
                G.append(
                    torch.cat(
                        [
                            (
                                p.grad.flatten()
                                if p.grad is not None
                                else torch.zeros(p.numel(), device=strategy.device)
                            )
                            for p in strategy.model.parameters()
                        ],
                        dim=0,
                    )
                )
            # Stack all experience gradient vectors into a matrix 
            self.G = torch.stack(G)  # (experiences, parameters)

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        # if we've already observed a task:
        if strategy.clock.train_exp_counter > 0:
            # iterate over all model parameters
            # for each: if gradient exists, flatten to a 1D tensor and if not then use a tensor of zeros of the same size
            g = torch.cat(
                [
                    (
                        p.grad.flatten()
                        if p.grad is not None
                        else torch.zeros(p.numel(), device=strategy.device)
                    )
                    for p in strategy.model.parameters()
                ],
                dim=0,
            )
            # if any gradients are negative then project
            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            # don't project (first experience)
            to_project = False
            # placeholder for v_star

        if to_project:
            # find closest vector to g

            # old code (exact, slower)
            # v_star = self.solve_quadprog(g).to(strategy.device)

            # new code (approximation, faster)
            v_star = self.solve_dualsgd(strategy.clock.train_exp_counter, strategy.device, g, 10)
            # print("V STAR SHAPE:", v_star.shape)
            
            # also old code
            num_pars = 0  # reshape v_star into the parameter matrices
            for p in strategy.model.parameters():
                curr_pars = p.numel()
                if p.grad is not None:
                    p.grad.copy_(v_star[num_pars : num_pars + curr_pars].view(p.size()))
                num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        """
        # update the memory buffer
        self.update_memory(
            strategy.experience.dataset,
            strategy.clock.train_exp_counter,
            strategy.train_mb_size,
        )

    @torch.no_grad()
    def update_memory(self, dataset, t, batch_size):
        """
        Update replay memory with patterns from current experience.
        """
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True
        )
        tot = 0
        for mbatch in dataloader:
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
            if tot + x.size(0) <= self.patterns_per_experience:
                if t not in self.memory_x:
                    self.memory_x[t] = x.clone()
                    self.memory_y[t] = y.clone()
                    self.memory_tid[t] = tid.clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid), dim=0)

            else:
                diff = self.patterns_per_experience - tot
                if t not in self.memory_x:
                    self.memory_x[t] = x[:diff].clone()
                    self.memory_y[t] = y[:diff].clone()
                    self.memory_tid[t] = tid[:diff].clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y[:diff]), dim=0)
                    self.memory_tid[t] = torch.cat(
                        (self.memory_tid[t], tid[:diff]), dim=0
                    )
                break
            tot += x.size(0)
    
    # t tasks, device dev, gradient g, iterations I
    def solve_dualsgd(self, t, dev, g, I):
        '''
        theory: v* <- 0-vector
        gradF w/ respect to v: G * (transpose(G) * v) + G * g
        new-v_star <- old-v_star - alpha * gradF
        new-v_star <- max[0-vector, v]
        '''
        # learning rate
        lr = 0.01

        # t may be exclusive, which means we would actually use t instead of t-1.         
        v = torch.zeros(t, device=dev)
        z = torch.zeros(t, device=dev)

        # does not depend on v_star
        # print(f"G shape: {self.G.shape}, g shape: {g.shape}")
        Gg = torch.mv(self.G, g)
        for _ in range(I):
            temp = torch.mv(self.G.T, v) # s ∈ n x 1
            full_product = torch.mv(self.G, temp)
            gradF = full_product + Gg
            v -= lr * gradF
            v = torch.max(v, z)
        # print(self.G.T.shape, v.shape, g.shape)
        # x = torch.mv(self.G.T, v)
        # print(x.shape)
        return torch.mv(self.G.T, v) + g
    

    # NOTE: commented out to reference later for matrix operations
    # def solve_quadprog(self, g):
    #     """
    #     Solve quadratic programming with current gradient g and
    #     gradients matrix on previous tasks G.
    #     Taken from original code:
    #     https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
    #     """

    #     memories_np = self.G.cpu().double().numpy()
    #     gradient_np = g.cpu().contiguous().view(-1).double().numpy()
    #     t = memories_np.shape[0]
    #     P = np.dot(memories_np, memories_np.transpose())
    #     P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
    #     q = np.dot(memories_np, gradient_np) * -1
    #     G = np.eye(t)
    #     h = np.zeros(t) + self.memory_strength
    #     # solution with old quadprog library, same as the author's implementation
    #     # v = quadprog.solve_qp(P, q, G, h)[0]
    #     # using new library qpsolvers
    #     v = qpsolvers.solve_qp(P=P, q=-q, G=-G.transpose(), h=-h, solver="quadprog")
    #     v_star = np.dot(v, memories_np) + gradient_np

    #     return torch.from_numpy(v_star).float()

