import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--h_parallel', type=int, default=1)
parser.add_argument('--g_parallel', type=int, default=1)
parser.add_argument('--d_parallel', type=int, default=1)
parser.add_argument('--block', type=str)
args = parser.parse_args()

n_heads = 32
hidden = 96 * n_heads
vocab_size = 50257
seq_len = 1024
batch_size = 512

head_parallel_x = args.h_parallel
gemm_parallel_x = args.g_parallel
data_parallel_x = args.d_parallel

model = nn.TransformerEncoderLayer(hidden, n_heads // head_parallel_x, 4*hidden // gemm_parallel_x)
mlp = nn.Sequential(*list(model.children())[1:])
attn = list(model.children())[0]

tt = []

if args.block == 'attn':
    model = attn.cuda()
    for i in range(100):
        src = torch.rand(seq_len, batch_size // data_parallel_x, hidden).cuda()
        with profile(activities=[
                ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("forward_pass"):
                output = model(src, src, src)
        tt.append(float(prof.key_averages().total_average().self_cuda_time_total_str[:-2]))
#        print(torch.cuda.memory_allocated())

elif args.block == 'mlp':
    model = mlp.cuda()
    for i in range(100):
        src = torch.rand(seq_len, batch_size // data_parallel_x, hidden).cuda()
        with profile(activities=[
                ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("forward_pass"):
                output = model(src)
        tt.append(float(prof.key_averages().total_average().self_cuda_time_total_str[:-2]))
#        print(torch.cuda.memory_allocated())

seq_proc = 100 * batch_size / data_parallel_x
tokens_proc = seq_proc * seq_len
print(1000 * tokens_proc / sum(tt))



#src = torch.randn(seq_len, batch_size, vocab_size)
#input = embedding(src) # vocab_size x hidden matrix
#output = model(input)
#output_seq = embedding(output) # hidden x vocab_size matrix
