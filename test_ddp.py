import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

rank = int(os.environ["LOCAL_RANK"])
print(f"rank = {rank}")

if rank != -1:
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl')