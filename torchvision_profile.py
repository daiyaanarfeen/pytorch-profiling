import argparse
import os
import torch
import torchvision.models as models
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.profiler

def init_process(fn, args):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend='nccl', init_method='env://')
    fn(args)
    dist.destroy_process_group()


def run(args):
    model = getattr(models, args.model)().cuda()
    rank = dist.get_rank()
    ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    inputs = torch.randn(args.bs, 3, 224, 224).cuda()
    labels = torch.randn(args.bs, 1000).cuda()

    tt = []
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=20, active=80, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.log_dir),
            record_shapes=True,
            with_stack=True
    ) as prof:
        for i in range(100):
            optimizer.zero_grad()
            preds = ddp_model(inputs)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            prof.step()
    print('success')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--bs', type=int)
    parser.add_argument('--model', type=str)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    parser.add_argument("--world_size", type=str)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--rank", type=str)
    parser.add_argument("--device", type=int)
    parser.add_argument("--log_dir", type=str)
    args = parser.parse_args()

#    os.environ["MASTER_ADDR"] = args.master_addr
#    os.environ["MASTER_PORT"] = args.master_port
#    os.environ["WORLD_SIZE"] = args.world_size
#    os.environ["RANK"] = args.rank

    torch.cuda.set_device(args.local_rank)
    init_process(run, args)
