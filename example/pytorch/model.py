###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from primuspipe.handler.wgrad_handler import WGRAD_RUNNING_CACHE


class LinearWithWgradFunc(torch.autograd.Function):
    """Custom autograd function that separates input gradient (dgrad) from weight gradient (wgrad).

    During backward, dgrad is computed immediately while wgrad is deferred as a
    closure in WGRAD_RUNNING_CACHE. This enables ZeroBubble / ZBV schedules to
    execute B (dgrad) and W (wgrad) at different time steps.
    """

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return torch.nn.functional.linear(input, weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        if weight.grad is None:
            weight.grad = torch.zeros_like(weight)

        def grad_weight_fn():
            weight.grad += grad_output.flatten(0, -2).T @ input.flatten(0, -2)

        WGRAD_RUNNING_CACHE.append(grad_weight_fn)

        grad_input = grad_output @ weight
        return grad_input, None


class LinearWithWgrad(nn.Module):
    """Drop-in replacement for nn.Linear that defers weight gradient computation."""

    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return LinearWithWgradFunc.apply(x, self.weight)


class MNISTStageModel(nn.Module):
    """A single pipeline stage for MNIST classification.

    The complete pipeline forms a multi-layer perceptron split across ranks:
        First stage:         Linear(784, hidden_size) + ReLU
        Intermediate stages: Linear(hidden_size, hidden_size) + ReLU
        Last stage:          Linear(hidden_size, 10) + CrossEntropyLoss
    """

    def __init__(
        self, is_first_stage: bool, is_last_stage: bool, hidden_size: int = 256, split_wgrad: bool = False
    ):
        super().__init__()
        in_features = 784 if is_first_stage else hidden_size
        out_features = 10 if is_last_stage else hidden_size
        if split_wgrad:
            self.fc = LinearWithWgrad(in_features, out_features)
        else:
            self.fc = nn.Linear(in_features, out_features)
        self.is_last_stage = is_last_stage

    def forward(self, x, y=None):
        x = self.fc(x)
        if not self.is_last_stage:
            x = F.relu(x)
        if y is not None:
            return F.cross_entropy(x, y)
        return x


def create_stage_model(
    pp_rank: int,
    pp_world_size: int,
    chunk: int = 0,
    vpp_size: int = 1,
    hidden_size: int = 256,
    vfold: bool = False,
    split_wgrad: bool = False,
):
    """Create a pipeline stage model for the given rank and chunk.

    For vpp_size=1 (basic 1F1B / ZeroBubble), each rank holds one stage.
    For vpp_size>1, each rank holds multiple stages (chunks).

    Topologies:
        Standard (interleaved):
            rank0-c0 -> rank1-c0 -> ... -> rankN-c0 -> rank0-c1 -> ... -> rankN-c1
            First = rank0-c0, Last = rankN-c(vpp-1)
        V-fold (ZBV):
            rank0-c0 -> rank1-c0 -> ... -> rankN-c0 -> rankN-c1 -> ... -> rank0-c1
            First = rank0-c0, Last = rank0-c1
    """
    is_first = pp_rank == 0 and chunk == 0
    if vfold:
        is_last = pp_rank == 0 and chunk == vpp_size - 1
    else:
        is_last = pp_rank == pp_world_size - 1 and chunk == vpp_size - 1
    return MNISTStageModel(is_first, is_last, hidden_size, split_wgrad=split_wgrad).cuda()


def load_mnist(batch_size: int, train: bool = True, data_root: str = "./data"):
    """Load MNIST dataset. All ranks must call this to stay in sync."""
    import torch.distributed as dist
    from torchvision import datasets, transforms

    if dist.is_initialized():
        if dist.get_rank() == 0:
            datasets.MNIST(data_root, train=train, download=True)
        dist.barrier()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.MNIST(data_root, train=train, download=True, transform=transform)

    g = torch.Generator().manual_seed(42)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, drop_last=True, generator=g
    )
