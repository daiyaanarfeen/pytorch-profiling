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
from deepspeech2 import DeepSpeech2

""" All-Reduce example."""
def run(log_dir, batch_size, model):
    """ Simple collective communication. """
    rank = dist.get_rank()
    bs = batch_size[0] if len(batch_size) == 1 else batch_size[rank]

    if model == 'resnet50':
        model = torchvision.models.resnet50()
        loss_fn = nn.MSELoss()
        inputs = torch.randn(bs, 3, 224, 224).cuda()
        labels = torch.randn(bs, 1000).cuda()

        grad_calc = lambda: loss_fn(ddp_model(inputs), labels).backward()
    elif model == 'deepspeech2':
        model = DeepSpeech2(num_classes=10, input_dim=80)
        criterion = nn.CTCLoss(blank=3, zero_infinity=True)
        input_lengths = np.random.normal(203021.64294474229, 57193.452740676534, 100).astype(int)
        target_lengths = np.random.normal(184.65321139493324, 58.645652429691275, 100).astype(int)
        inputs = torch.rand(bs, max(input_lengths), 80).cuda()
        targets = torch.randint(0, 10, [bs, max(target_lengths)]).cuda()

        forward = lambda: ddp_model(inputs, input_lengths)
        loss = lambda outputs: criterion(outputs[0], targets, outputs[1], target_lengths)
        grad_calc = lambda: loss(forward()).backward()
    elif model == 'yolov3':
        pass
    elif model == 'ncf':
        pass

    model.cuda()
    ddp_model = DDP(model, broadcast_buffers=False, bucket_cap_mb=25)
    optimizer = optim.AdamW(ddp_model.parameters())


    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=5, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            with_stack=True
    ) as prof:
        for i in range(10):
            optimizer.zero_grad()
            grad_calc()
            optimizer.step()
            prof.step()

    print('success')


def init_process(fn, log_dir, batch_size, model):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend='nccl', init_method='env://')
    fn(log_dir, batch_size, model)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=str)
    parser.add_argument("--world_size", type=str)
    parser.add_argument("--rank", type=str)
    parser.add_argument("--device", type=int)
    parser.add_argument("--batch_size", type=int, nargs="+") # always a list
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["WORLD_SIZE"] = args.world_size
    os.environ["RANK"] = args.rank

    torch.cuda.set_device(args.device)
    init_process(run, args.log_dir, args.batch_size, args.model)
