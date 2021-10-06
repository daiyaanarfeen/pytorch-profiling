import argparse
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--bs', type=int)
args = parser.parse_args()

model = models.resnet18().cuda()
tt = []
for i in range(100):
    inputs = torch.randn(args.bs, 3, 224, 224).cuda()
    
    with profile(activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    
    tt.append(float(prof.key_averages().total_average().self_cuda_time_total_str[:-2]))
print(sum(tt) / len(tt))
