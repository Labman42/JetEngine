from accelerate import PartialState
import torch.distributed as dist


class DistributedManager:
    """
    A manager for the distributed environment, using accelerate.PartialState.
    """

    def __init__(self, tp_size: int):
        self.state = PartialState()
        self.tp_size = tp_size
        self.dp_size = self.state.num_processes // self.tp_size
        self.tp_group = None
        self.dp_group = None

        # Create tensor parallel (TP) process groups
        for i in range(self.dp_size):
            ranks = list(range(i * self.tp_size, (i + 1) * self.tp_size))
            group = dist.new_group(ranks)
            if self.state.process_index in ranks:
                self.tp_group = group
                self.tp_rank = self.state.process_index % self.tp_size

        # Create data parallel (DP) process groups
        for i in range(self.tp_size):
            ranks = list(range(i, self.state.num_processes, self.tp_size))
            group = dist.new_group(ranks)
            if self.state.process_index in ranks:
                self.dp_group = group
                self.dp_rank = self.state.process_index // self.tp_size

    @property
    def device(self):
        return self.state.device

    def wait_for_everyone(self):
        self.state.wait_for_everyone()
