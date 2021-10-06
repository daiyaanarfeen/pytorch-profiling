import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

""" All-Reduce example."""
def run():
    """ Simple collective communication. """
#    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print('Rank ', dist.get_rank(), ' has data ', tensor[0])

def init_process(fn, backend='gloo'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend='nccl', init_method='env://')
    fn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    init_process(run)
