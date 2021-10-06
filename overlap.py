import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

""" All-Reduce example."""
def run():
    """ Simple collective communication. """
    rank = dist.get_rank()
    model = 
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()

    outputs = ddp_model()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    print('Rank ', dist.get_rank(), ' has data ', tensor[0])

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
