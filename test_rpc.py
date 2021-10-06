import torch
import torch.distributed.rpc as rpc
import torch.autograd.profiler as profiler
import torch.multiprocessing as mp
import os
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

def random_tensor(size):
    size = size * 1024**2
    return torch.rand((size), requires_grad=False)


def return_tensor():
    global tensor
    return tensor


def worker(world_size, data_size):
#    os.environ["MASTER_ADDR"] = "localhost"
#    os.environ["MASTER_PORT"] = "29500"
#    worker_name = f"worker{rank}"

    # Initialize RPC framework.
    torch.distributed.init_process_group(backend='gloo', init_method='env://')
    rank = dist.get_rank()
    worker_name = f"worker{rank}"
    rpc.init_rpc(
        name=worker_name,
        rank=rank,
        world_size=world_size
    )
    logger.debug(f"{worker_name} successfully initialized RPC.")

    global tensor
    tensor = random_tensor(data_size)

    if rank == 0:
        print(tensor)
        dst_worker_rank = (rank + 1) % world_size
        dst_worker_name = f"worker{dst_worker_rank}"
#        t1, t2 = random_tensor(), random_tensor()
        # Send and wait RPC completion under profiling scope.
        with profiler.profile() as prof:
            fut1 = rpc.rpc_sync(dst_worker_name, return_tensor, args=())
            print(fut1)
#            fut2 = rpc.rpc_async(dst_worker_name, torch.mul, args=(t1, t2))
            # RPCs must be awaited within profiling scope.
#            fut1.wait()
#            fut2.wait()

        print(prof.key_averages().table())

    logger.debug(f"Rank {rank} waiting for workers and shutting down RPC")
    rpc.shutdown()
    logger.debug(f"Rank {rank} shutdown RPC")


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int)
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    # Run 2 RPC workers.
#    world_size = 2
#    mp.spawn(worker, args=(world_size, args.data_size,), nprocs=world_size)
    worker(2, args.data_size)
