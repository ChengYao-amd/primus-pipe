###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MNIST Training with ZBV Formatted Pipeline Parallelism

ZBV (Zero Bubble V-shape) uses a V-fold topology where the data flows forward
through ranks 0->1->...->N-1, then folds back N-1->...->0. This means rank 0
is both the first and last virtual stage. Combined with B/W split scheduling,
this achieves near-zero pipeline bubbles.

Forward path:  rank0(c0) -> rank1(c0) -> ... -> rankN(c0)
                  -> rankN(c1) -> ... -> rank1(c1) -> rank0(c1)

Usage:
    torchrun --nproc_per_node=4 example/pytorch/zbv_formatted_demo.py
"""

import functools
from typing import Any, List

import torch
from dist_manager import DistManager
from model import create_stage_model, load_mnist

from primuspipe.handler import default_handler_dict
from primuspipe.scheduler.algorithms.zbv_formatted import ScheduleZBVFormatted
from primuspipe.scheduler.scheduler import ScheduleRunner
from primuspipe.scheduler.scheduler_node import FuncType, SchedulerNode

_input_cache: dict[int, torch.Tensor] = {}
_step_losses: list[float] = []


def forward_step(
    models: dict[int, torch.nn.Module],
    dist_manager: DistManager,
    conf_dict: dict[str, Any],
    micro_images: dict[int, torch.Tensor],
    micro_labels: dict[int, torch.Tensor],
    input_tensor,
    minibatch: int,
    chunk: int,
):
    pp_rank = dist_manager.get_pp_rank()
    model = models[chunk]

    is_first = pp_rank == 0 and chunk == 0
    is_last = pp_rank == 0 and chunk == 1

    if is_first:
        x = micro_images[minibatch].clone().requires_grad_(True)
        _input_cache[minibatch] = x
    else:
        x = input_tensor

    y = micro_labels[minibatch] if is_last else None
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

    is_first = pp_rank == 0 and chunk == 0
    is_last = pp_rank == 0 and chunk == 1

    if is_last:
        output_tensors.backward()
        output_tensors.detach_()
    else:
        torch.autograd.backward(output_tensors, grad_tensors)

    if is_first:
        input_tensors = _input_cache.pop(minibatch)

    assert input_tensors is not None
    return input_tensors.grad.clone().detach()


def bind_func_for_scheduler_table(
    scheduler_table: List[List[SchedulerNode]],
    dist_manager: DistManager,
    conf_dict: dict[str, Any],
    models: dict[int, torch.nn.Module],
    micro_images: dict[int, torch.Tensor],
    micro_labels: dict[int, torch.Tensor],
):
    pp_rank = dist_manager.get_pp_rank()
    my_schedule = scheduler_table[pp_rank]

    fwd_func = functools.partial(forward_step, models, dist_manager, conf_dict, micro_images, micro_labels)
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


def run_model(pp_size: int, vpp_size: int = 2):
    dist_manager = DistManager(pp_size=pp_size)
    pp_rank = dist_manager.get_pp_rank()
    pp_world_size = dist_manager.get_pp_world_size()
    is_loss_rank = pp_rank == 0

    conf_dict = {
        "micro_batch_size": 64,
        "hidden_size": 256,
        "micro_batches": 8,
        "vpp_size": vpp_size,
        "communication_mode": "batch_p2p",
    }

    models = {
        c: create_stage_model(
            pp_rank,
            pp_world_size,
            chunk=c,
            vpp_size=vpp_size,
            hidden_size=conf_dict["hidden_size"],
            vfold=True,
            split_wgrad=True,
        )
        for c in range(vpp_size)
    }
    all_params = [p for m in models.values() for p in m.parameters()]
    optimizer = torch.optim.Adam(all_params, lr=1e-3)

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

            schedule = ScheduleZBVFormatted(pp_size=pp_world_size, vpp_size=vpp_size, micro_batches=n_mb)
            schedule_table = schedule.generate_schedule_table()
            bind_func_for_scheduler_table(
                schedule_table, dist_manager, conf_dict, models, micro_images, micro_labels
            )

            _step_losses.clear()
            schedule_runner.run(schedule_table, pp_rank)

            optimizer.step()
            optimizer.zero_grad()

            if is_loss_rank and step % 20 == 0:
                avg_loss = sum(_step_losses) / len(_step_losses) if _step_losses else 0
                print(f"[Epoch {epoch + 1}/{num_epochs}] Step {step:4d} | Loss: {avg_loss:.4f}")

    if is_loss_rank:
        print("Training complete!")

    dist_manager.cleanup()


if __name__ == "__main__":
    """torchrun --nproc_per_node=4 example/pytorch/zbv_formatted_demo.py"""
    run_model(pp_size=4, vpp_size=2)
