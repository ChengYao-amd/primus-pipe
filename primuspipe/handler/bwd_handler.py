###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

""" Handle for B (backward input gradient only, weight gradient deferred to W step) """

from ..scheduler.scheduler_node import FuncType, SchedulerNode
from .utils import find_prev_node_with_type
from .wgrad_handler import WGRAD_RUNNING_CACHE


def default_bwd_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):

    WGRAD_RUNNING_CACHE.set_current_minibatch_and_chunk(node.mini_batch, node.chunk)

    fwd_node_idx = find_prev_node_with_type(scheduler_table, idx, [FuncType.F])
    assert fwd_node_idx is not None
    output = scheduler_table[fwd_node_idx].args["output"]
    input = scheduler_table[fwd_node_idx].args["input"]
    recv_node_idx = find_prev_node_with_type(scheduler_table, idx, [FuncType.RB])

    output_grad = None

    if recv_node_idx is not None:
        if "req" in scheduler_table[recv_node_idx].args:
            scheduler_table[recv_node_idx].args["req"].wait()
            scheduler_table[recv_node_idx].args["req"] = None
            del scheduler_table[recv_node_idx].args["req"]

        output_grad = scheduler_table[recv_node_idx].args["recv_buffer"]

    input_grad = node.args["backward_func"](input, output, output_grad, node.mini_batch, node.chunk)
    node.args["output"] = input_grad

    scheduler_table[fwd_node_idx].args["output"] = None
    scheduler_table[fwd_node_idx].args["input"] = None
