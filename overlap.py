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
def run(log_dir, batch_size):
    """ Simple collective communication. """
    rank = dist.get_rank()
    model = torchvision.models.resnet18() 
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], broadcast_buffers=False, bucket_cap_mb=25)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    bs = batch_size[0] if len(batch_size) == 1 else batch_size[rank]
    inputs = torch.randn(bs, 3, 224, 224).to(rank)
    labels = torch.randn(bs, 1000).to(rank)


    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=5, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            with_stack=True
    ) as prof:
        for i in range(10):
            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss_fn(outputs, labels).backward()
            optimizer.step()
            prof.step()

    print('success')


def init_process(fn, log_dir, batch_size):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend='nccl', init_method='env://')
    fn(log_dir, batch_size)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    parser.add_argument("--world_size", type=str)
    parser.add_argument("--rank", type=str)
    parser.add_argument("--batch_size", type=int, nargs="+") # always a list
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["WORLD_SIZE"] = args.world_size
    os.environ["RANK"] = args.rank

    init_process(run, args.log_dir, args.batch_size)
