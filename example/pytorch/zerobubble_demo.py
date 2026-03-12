###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MNIST Training with Zero Bubble Pipeline Parallelism

Zero Bubble splits the backward pass into B (input gradient only) and W
(weight gradient), allowing more flexible scheduling that minimizes pipeline
bubbles. The schedule follows an F-B-W pattern where W can be deferred.

Usage:
    torchrun --nproc_per_node=4 example/pytorch/zerobubble_demo.py
"""

import functools
from typing import Any, List

import torch
from dist_manager import DistManager
from model import create_stage_model, load_mnist

from primuspipe.handler import default_handler_dict
from primuspipe.scheduler.algorithms.zerobubble import ScheduleZeroBubble
from primuspipe.scheduler.scheduler import ScheduleRunner
from primuspipe.scheduler.scheduler_node import FuncType, SchedulerNode

_input_cache: dict[int, torch.Tensor] = {}
_step_losses: list[float] = []


def forward_step(
    model,
    dist_manager: DistManager,
    conf_dict: dict[str, Any],
    micro_images: dict[int, torch.Tensor],
    micro_labels: dict[int, torch.Tensor],
    input_tensor,
    minibatch: int,
    chunk: int,
):
    pp_rank = dist_manager.get_pp_rank()
    pp_world_size = dist_manager.get_pp_world_size()

    if pp_rank == 0:
        x = micro_images[minibatch].clone().requires_grad_(True)
        _input_cache[minibatch] = x
    else:
        x = input_tensor

    y = micro_labels[minibatch] if pp_rank == pp_world_size - 1 else None
    output = model(x, y)

    if y is not None:
        _step_losses.append(output.item())
    return output


def backward_step(
    dist_manager: DistManager,
    input_tensors: torch.Tensor,
    output_tensors: torch.Tensor,
    grad_tensors: torch.Tensor,
    minibatch: int,
    chunk: int,
):
    pp_rank = dist_manager.get_pp_rank()
    pp_world_size = dist_manager.get_pp_world_size()

    if pp_rank == pp_world_size - 1:
        output_tensors.backward()
        output_tensors.detach_()
    else:
        torch.autograd.backward(output_tensors, grad_tensors)

    if pp_rank == 0:
        input_tensors = _input_cache.pop(minibatch)

    assert input_tensors is not None
    return input_tensors.grad.clone().detach()


def bind_func_for_scheduler_table(
    scheduler_table: List[List[SchedulerNode]],
    dist_manager: DistManager,
    conf_dict: dict[str, Any],
    model,
    micro_images: dict[int, torch.Tensor],
    micro_labels: dict[int, torch.Tensor],
):
    pp_rank = dist_manager.get_pp_rank()
    my_schedule = scheduler_table[pp_rank]

    fwd_func = functools.partial(forward_step, model, dist_manager, conf_dict, micro_images, micro_labels)
    bwd_func = functools.partial(backward_step, dist_manager)

    for node in my_schedule:
        if node.args is None:
            node.args = {}
        if node.meta is None:
            node.meta = {}

        if node.func_type == FuncType.F:
            node.args["fwd_func"] = fwd_func
        elif node.func_type == FuncType.B:
            node.args["backward_func"] = bwd_func
        elif node.func_type in [FuncType.RF, FuncType.RB]:
            node.args["recv_buffer_size"] = (conf_dict["micro_batch_size"], conf_dict["hidden_size"])
            node.args["dtype"] = torch.float32

        if node.func_type in [FuncType.SF, FuncType.SB, FuncType.RF, FuncType.RB]:
            node.args["pp_group"] = dist_manager.get_pp_group()
            node.meta["communication_mode"] = conf_dict["communication_mode"]


def run_model(pp_size: int):
    dist_manager = DistManager(pp_size=pp_size)
    pp_rank = dist_manager.get_pp_rank()
    pp_world_size = dist_manager.get_pp_world_size()
    is_last = pp_rank == pp_world_size - 1

    conf_dict = {
        "micro_batch_size": 64,
        "hidden_size": 256,
        "micro_batches": 8,
        "communication_mode": "batch_p2p",
    }

    model = create_stage_model(pp_rank, pp_world_size, hidden_size=conf_dict["hidden_size"], split_wgrad=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_batch = conf_dict["micro_batch_size"] * conf_dict["micro_batches"]
    train_loader = load_mnist(total_batch, train=True)

    schedule_runner = ScheduleRunner(default_handler_dict)

    num_epochs = 3
    for epoch in range(num_epochs):
        for step, (images, labels) in enumerate(train_loader):
            images = images.view(images.size(0), -1).cuda()
            labels = labels.cuda()

            mb = conf_dict["micro_batch_size"]
            n_mb = conf_dict["micro_batches"]
            micro_images = {i: images[i * mb : (i + 1) * mb] for i in range(n_mb)}
            micro_labels = {i: labels[i * mb : (i + 1) * mb] for i in range(n_mb)}

            schedule = ScheduleZeroBubble(pp_size=pp_world_size, vpp_size=1, micro_batches=n_mb)
            schedule_table = schedule.generate_schedule_table()
            bind_func_for_scheduler_table(
                schedule_table, dist_manager, conf_dict, model, micro_images, micro_labels
            )

            _step_losses.clear()
            schedule_runner.run(schedule_table, pp_rank)

            optimizer.step()
            optimizer.zero_grad()

            if is_last and step % 20 == 0:
                avg_loss = sum(_step_losses) / len(_step_losses) if _step_losses else 0
                print(f"[Epoch {epoch + 1}/{num_epochs}] Step {step:4d} | Loss: {avg_loss:.4f}")

    if is_last:
        print("Training complete!")

    dist_manager.cleanup()


if __name__ == "__main__":
    """torchrun --nproc_per_node=4 example/pytorch/zerobubble_demo.py"""
    run_model(pp_size=4)
