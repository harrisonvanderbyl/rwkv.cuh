import safetensors.torch as st
import torch
import os

# change path to your model


from torch.utils.cpp_extension import load
quant_cpp = load(name="quant_cpp", sources=["./quant.cpp"], verbose=True,
        extra_cflags=["-O3", "-march=native"  ,"-flto", "-funroll-loops", "-D_GLIBCXX_PARALLEL"])


import inquirer

questions = [
    inquirer.List('file',
                message="What file do you want to convert?",
                choices=[file for file in os.listdir("./") if ".pth" in file],
            )
    ,
    inquirer.List('mode',
                message="What do you want to do?",
                choices=['Convert to BF16', 'Convert to FP32', 'Convert to Uint8'],
            ),
]
resp = inquirer.prompt(questions)
path = resp["file"]
model = torch.load(path, "cpu")
mode = resp["mode"]
bf16 = mode == "Convert to BF16"
fp32 = mode == "Convert to FP32"
uint8 = mode == "Convert to Uint8"


# float32= Tensor(-0.477592, -9.87757, -10.544, -9.66997, , ..., -23.7958, -23.7907, -23.739, -23.7474, shape=(1, 18, 65536))


import tqdm as tqdm
keys = [*model.keys()]
for key in tqdm.tqdm(keys):
    if model[key].shape.__len__() == 2 and key != "emb.weight" and "time_" not in key:
        
        # bf16 conversion for avx512
        if bf16:
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
            model[key] = model[key].bfloat16().clone().cpu()
        else:
            model[key] = model[key].float().cpu()
    else:
        if bf16:
            model[key] = model[key].bfloat16().clone().cpu()
        else:
            model[key] = model[key].float().clone().cpu()
    if "decay" in key:
        if bf16:
            model[key] = model[key].double().exp().neg().exp().bfloat16().cpu()
        else:
            model[key] = model[key].double().exp().neg().exp().float().cpu()
# create ../build/ if not exists
import os

st.save_file(model, "../model.safetensors")