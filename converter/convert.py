import safetensors.torch as st
import torch


# change path to your model
path = "./1b6v6.pth"
model = torch.load(path, map_location=torch.device('cpu'))

from torch.utils.cpp_extension import load
quant_cpp = load(name="quant_cpp", sources=["./quant.cpp"], verbose=True,
        extra_cflags=["-O3", "-march=native"  ,"-flto", "-funroll-loops", "-D_GLIBCXX_PARALLEL"])


import inquirer
modeenum = [
    "Convert to BF16",
    "Convert to FP32",
    "Convert to Uint8",
    "Convert to BF16UINT8"
]

questions = [
    inquirer.List('mode',
                message="What do you want to do?",
                choices=modeenum
            ),
]
mode = inquirer.prompt(questions)["mode"]
bf16 = False
uint8 = False

print(mode)

if mode == modeenum[0]:
    bf16 = True
elif mode == modeenum[1]:
    bf16 = False
elif mode == modeenum[2]:
    uint8 = True
elif mode == modeenum[3]:
    bf16 = True
    uint8 = True

print("Converting to ", "BF16" if bf16 else "FP32", "and Uint8" if uint8 else "")
# float32= Tensor(-0.477592, -9.87757, -10.544, -9.66997, , ..., -23.7958, -23.7907, -23.739, -23.7474, shape=(1, 18, 65536))


import tqdm as tqdm
keys = [*model.keys()]
v6 = False
if("blocks.1.att.time_decay_w1" in keys):
    v6 = True 
for key in tqdm.tqdm(keys):
    if model[key].shape.__len__() == 2 and key != "emb.weight" and "time_" not in key:
        
        # bf16 conversion for avx512
        if bf16 and not uint8:
            
            model[key] = model[key].bfloat16().clone().cpu().contiguous()
            shape = model[key].shape
            # model[key] = model[key].reshape(-1,2,16)[:,[1,0]].reshape(shape)
        else:
            if uint8:
                weight = model[key].t().float().clone().cpu()
                model[key] = (torch.zeros(weight.shape[1],weight.shape[0]).to(torch.uint8))
                model[key+".range"] = (torch.zeros(weight.shape[1]))
                model[key+".zero"] = (torch.zeros(weight.shape[1]))
                quant_cpp.quantize_cpu(weight.t().contiguous(), model[key+".range"] , model[key+".zero"],model[key], weight.shape[1], weight.shape[0], True)
                # model[key] = model[key].t().contiguous().cpu()
            else:
                model[key] = model[key].float().clone().cpu()
                
    elif model[key].shape.__len__() == 1:
        if bf16:
            model[key] = model[key].bfloat16().clone().cpu().contiguous()
        else:
            model[key] = model[key].float().cpu().contiguous()
    else:
        if bf16:
            model[key] = model[key].bfloat16().clone().cpu().contiguous()
        else:
            model[key] = model[key].float().clone().cpu().contiguous()\
                
    if "maa_w1" in key or "decay_w1" in key or "decay_w2" in key:
        # transpose
        print("tansposing:"+key)
        model[key] = model[key].t().contiguous()
    if "maa_w2" in key:
        #transpose last 2 dims
        print("tansposing:"+key)
        model[key] = model[key].transpose(-1,-2).contiguous()
    if "decay" in key and not v6:
        print(key)
        if bf16:
            model[key] = model[key].double().exp().neg().exp().bfloat16().cpu()
        else:
            model[key] = model[key].double().exp().neg().exp().float().cpu()
            
# create ../build/ if not exists
import os

st.save_file(model, "../model.safetensors")