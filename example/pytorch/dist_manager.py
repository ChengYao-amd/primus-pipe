###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os

import torch
import torch.distributed as dist


class DistManager:
    def __init__(self, pp_size):
        """assert only PP + DP is supported"""
        self.pp_size = pp_size

        if not dist.is_initialized():
            self.rank = int(os.environ.get("RANK", 0))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))

            dist.init_process_group(backend="nccl")

            assert self.world_size % self.pp_size == 0, "world_size must be divisible by pp_size"
            self.dp_size = self.world_size // self.pp_size
            self.dp_rank = self.rank // self.pp_size

            for dp_replica in range(self.dp_size):
                pp_ranks = [dp_replica * self.pp_size + i for i in range(self.pp_size)]
                pp_group = dist.new_group(ranks=pp_ranks)

                if self.rank in pp_ranks:
                    self.pp_group = pp_group

            for i in range(self.pp_size):
                dp_ranks = [j * self.pp_size + i for j in range(self.dp_size)]
                dp_group = dist.new_group(ranks=dp_ranks)

                if self.rank in dp_ranks:
                    self.dp_group = dp_group

            torch.cuda.set_device(self.local_rank)

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size

    def get_dp_rank(self):
        return self.dp_group.rank()

    def get_dp_world_size(self):
        return self.dp_group.size()

    def get_dp_group(self):
        return self.dp_group

    def get_pp_rank(self):
        return self.pp_group.rank()

    def get_pp_world_size(self):
        return self.pp_group.size()

    def get_pp_group(self):
        return self.pp_group

    def cleanup(self):
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    torchrun --nproc_per_node=8 dist_manager.py
    """

    pp_size = 4
    dist_manager = DistManager(pp_size=pp_size)

    print(
        f"rank: {dist_manager.get_rank()}, world_size: {dist_manager.get_world_size()}, dp_rank: {dist_manager.get_dp_rank()}, dp_world_size: {dist_manager.get_dp_world_size()}, pp_rank: {dist_manager.get_pp_rank()}, pp_world_size: {dist_manager.get_pp_world_size()}"
    )

    dist_manager.cleanup()
