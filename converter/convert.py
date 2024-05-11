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
                choices=['Convert to FP32', 'Convert to Uint8'],
            ),
]
resp = inquirer.prompt(questions)
path = resp["file"]
model = torch.load(path, "cpu")
mode = resp["mode"]
fp32 = mode == "Convert to FP32"
uint8 = mode == "Convert to Uint8"


# float32= Tensor(-0.477592, -9.87757, -10.544, -9.66997, , ..., -23.7958, -23.7907, -23.739, -23.7474, shape=(1, 18, 65536))
def get_model_layout(torch_weights):
    
    vocab_size, n_embd = torch_weights["emb.weight"].shape
    
    dim_ffn = torch_weights[f"blocks.0.ffn.value.weight"].shape[1]
  
    n_head = torch_weights[f"blocks.0.att.time_decay"].shape[0]
    
    headsize = n_embd // n_head
    
    n_layer = len([x for x in torch_weights.keys() if x.startswith("blocks.") and x.endswith(".att.time_decay")])
    
    return n_layer, n_embd, vocab_size, headsize, dim_ffn, n_head


import tqdm as tqdm
keys = [*model.keys()]
for key in tqdm.tqdm(keys):
    if model[key].shape.__len__() == 2 and key != "emb.weight" and "time_" not in key:
        
        # bf16 conversion for avx512
        
        if uint8:
            ww = model.pop(key).t()
            weight = ww.float().clone().cpu()
            
            model[key] = (torch.zeros(weight.shape[1],weight.shape[0]).to(torch.uint8))
            model[key+".range"] = (torch.zeros(weight.shape[1]))
            model[key+".zero"] = (torch.zeros(weight.shape[1]))
            
            #if("output" not in key and 'ffn.receptance' not in key):
            mww = weight.t().contiguous()
            
            quant_cpp.quantize_cpu(mww, model[key+".range"] , model[key+".zero"],model[key], weight.shape[1], weight.shape[0], True)
            
            
            # model[key] = model.pop(key).t().contiguous().cpu()
        else:
            model[key] = model.pop(key).float().clone().cpu()
                
    elif model[key].shape.__len__() == 1:
        model[key] = model.pop(key).float().cpu()
    else:
        model[key] = model.pop(key).float().clone().cpu()
    if "decay" in key:
        model[key] = model.pop(key).double().exp().neg().exp().float().cpu()
    
n_layer, n_embd, vocab_size, headsize, dim_ffn, n_head = get_model_layout(model)

for i in range(n_layer):
    model[f"blocks.{i}.attshift.time_mix"] = torch.stack([model.pop(f"blocks.{i}.att.time_mix_{j}") for j in "krvg"]).reshape(4,-1)
    model[f"blocks.{i}.ffnshift.time_mix"] = torch.stack([model.pop(f"blocks.{i}.ffn.time_mix_{j}") for j in "kr"]).reshape(2,-1)
# create ../build/ if not exists
import os

st.save_file(model, "../model.safetensors")