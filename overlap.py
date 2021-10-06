import os
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.profiler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

""" All-Reduce example."""
def run():
    """ Simple collective communication. """
    rank = dist.get_rank()
    model = torchvision.models.resnet18() 
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    inputs = torch.randn(32, 3, 224, 244).to(rank)
    labels = torch.randn(32, 1000).to(rank)

    with torch.profiler.profile(
#            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
            record_shapes=True,
            with_stack=True
    ) as prof:
        outputs = ddp_model(inputs)
        loss_fn(outputs, labels).backward()
        optimizer.step()
        prof.step()

    print('success')


def init_process(fn, backend='gloo'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend='nccl', init_method='env://')
    fn()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    init_process(run)
