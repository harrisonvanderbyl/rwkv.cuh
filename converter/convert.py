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

    is_v6 =f"blocks.0.att.time_decay_w2" in torch_weights.keys()
    
    return n_layer, n_embd, vocab_size, headsize, dim_ffn, n_head, is_v6


import tqdm as tqdm
keys = [*model.keys()]
print(keys)
for key in tqdm.tqdm(keys):
    if model[key].shape.__len__() == 2 and key != "emb.weight" and "time_" not in key and "weight" in key:
        
        # bf16 conversion for avx512
        
        if uint8:
            ww = model.pop(key).t()
            weight = ww.float().clone().cpu()
            
            model[key] = (torch.zeros(weight.shape[1],weight.shape[0]).to(torch.uint8))
            model[key+".range"] = (torch.zeros(weight.shape[1], dtype=torch.float64))
            model[key+".zero"] = (torch.zeros(weight.shape[1], dtype=torch.float64))
            
            #if("output" not in key and 'ffn.receptance' not in key):
            mww = weight.t().contiguous()
            
            quant_cpp.quantize_cpu(mww, model[key+".range"] , model[key+".zero"],model[key], weight.shape[1], weight.shape[0], True)
            
            model[key+".range"]  = model[key+".range"].double()
            model[key+".zero"]  = model[key+".zero"].double()
            
            # model[key] = model.pop(key).t().contiguous().cpu()
        else:
            model[key] = model.pop(key).float().clone().cpu()
                
    elif model[key].shape.__len__() == 1:
        model[key] = model.pop(key).float().cpu()
    else:
        model[key] = model.pop(key).float().clone().cpu()
    
    if("time_mix" in key):
        model[key] = 1.0 - model.pop(key)
   
    
n_layer, n_embd, vocab_size, headsize, dim_ffn, n_head, is_v6 = get_model_layout(model)
zm1 = torch.zeros(0,n_embd)
zm2 = torch.zeros(n_embd,0)
zm1w = torch.zeros(1,n_embd)
zm2w = torch.zeros(n_embd,1)
zm4 = torch.zeros(5,n_embd,0)
zm3 = torch.zeros(2,n_embd,0)
for i in range(n_layer):

    model[f"blocks.{i}.ffnshift.time_mix_x"] = torch.zeros(1,n_embd)
    model[f"blocks.{i}.ffnshift.time_mix_w1.weight"] = zm1
    model[f"blocks.{i}.ffnshift.time_mix_w2.weight"] = zm3
    maamix = "maa" if is_v6 else "mix"
    model[f"blocks.{i}.ffnshift.time_mix_w2.bias"] = torch.stack([model.pop(f"blocks.{i}.ffn.time_{maamix}_{j}") for j in "kr"]).reshape(2,-1)
    model[f"blocks.{i}.att.w2.bias"] = model.pop(f"blocks.{i}.att.time_decay").reshape(n_embd).float().cpu()
    if(not is_v6):
        model[f"blocks.{i}.att.w2.bias"] = model[f"blocks.{i}.att.w2.bias"].double().exp().neg().exp().float()

    model[f"blocks.{i}.att.w1.weight"] = model.pop(f"blocks.{i}.att.time_decay_w1").t().contiguous() if is_v6 else zm1
    model[f"blocks.{i}.att.w2.weight"] = model.pop(f"blocks.{i}.att.time_decay_w2").t().contiguous() if is_v6 else zm2
    model[f"blocks.{i}.attshift.time_mix_x"] = model.pop(f"blocks.{i}.att.time_maa_x").reshape(1,-1) if is_v6 else torch.zeros(1,n_embd)
    model[f"blocks.{i}.attshift.time_mix_w1.weight"] = model.pop(f"blocks.{i}.att.time_maa_w1").transpose(0,1).contiguous() if is_v6 else zm1
    model[f"blocks.{i}.attshift.time_mix_w2.weight"] = model.pop(f"blocks.{i}.att.time_maa_w2").transpose(1,2).contiguous() if is_v6 else zm4
    model[f"blocks.{i}.attshift.time_mix_w2.bias"] = torch.stack(([model.pop(f"blocks.{i}.att.time_{maamix}_{j}") for j in ("wkvrg" if is_v6 else "kvrg") ]*2)[-5:]).reshape(5,-1)
    

# create ../build/ if not exists
import os

st.save_file(model, "../model.safetensors")