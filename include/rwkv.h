#include <iostream>
#include <string>
#include <fstream>
#include "tensor/safetensors.h"
#include "tensor/modules/embedding.h"
#include "tensor/modules/layernorm.h"
#include "tensor/modules/linear.h"
#include "tensor/modules/block.h"

class RWKV
{

public:
    Embedding emb1;
    LayerNorm ln0;
    LayerNorm ln_out;
    Linear output;
    std::vector<Block> blocks;
    safetensors model;

    size_t layers;
    size_t max_batch_seq = 0;

    RWKV(std::string path)
    {
        std::ifstream inFile;
        inFile.open(path, std::ios::binary);
        model = safetensors(inFile);
        

        auto keys = model.keys();
        layers = 0;
        for (auto key : keys)
        {
            if (std::string(key).find("blocks.") != std::string::npos)
            {
                if (std::string(key).find("att.time_mix_k") != std::string::npos)
                {
                    layers++;
                }
               
            }
        }

        // std::cout << "layers:" << layers << std::endl;

        auto t1o = model["emb.weight"];
        this->emb1 = Embedding(t1o);
        this->ln0 = LayerNorm(model["blocks.0.ln0.weight"], model["blocks.0.ln0.bias"]);
        this->ln_out = LayerNorm(model["ln_out.weight"], model["ln_out.bias"]);
        this->output = Linear(model, "head");
        for (size_t i = 0; i < layers; i++)
        {
            blocks.push_back(Block(model, i));
        }
    }

    Tensor operator()(std::vector<std::vector<size_t>> input)
    {
        auto x = emb1(input);
        x = ln0(x);
        for (size_t i = 0; i < layers; i++)
        {
            x = blocks[i](x);

        }
        auto xm = ln_out(x);
        return output(xm);

    }

    void get_state(std::map<std::string, Tensor> state, size_t batchid = 0){
       
        
        for (size_t i = 0; i < layers; i++)
        {
            auto wkv = blocks[i].att.state[batchid];
            auto ts1 = blocks[i].att.timeshift.state[batchid];
            auto ts2 = blocks[i].ffn.timeshift.state[batchid];
          
            state["blocks." + std::to_string(i) + ".att.state"].copyfrom(wkv);
            state["blocks." + std::to_string(i) + ".att.timeshift.state"].copyfrom(ts1);
            state["blocks." + std::to_string(i) + ".ffn.timeshift.state"].copyfrom(ts2);
            
        }
    }

    void set_state(std::map<std::string, Tensor> state, size_t batchid = 0){
        for (size_t i = 0; i < layers; i++)
        {
            auto wkv = state["blocks." + std::to_string(i) + ".att.state"];
            auto ts1 = state["blocks." + std::to_string(i) + ".att.timeshift.state"];
            auto ts2 = state["blocks." + std::to_string(i) + ".ffn.timeshift.state"];

            // std::cout << "wkv:" << wkv.shape[0] << " : " << wkv.shape[1] << std::endl;
            // std::cout << "ts1:" << ts1.shape[0] << " : " << ts1.shape[1] << std::endl;
            // std::cout << "ts2:" << ts2.shape[0] << " : " << ts2.shape[1] << std::endl;
            blocks[i].att.state[batchid].copyfrom(wkv);
            blocks[i].att.timeshift.state[batchid].copyfrom(ts1);
            blocks[i].ffn.timeshift.state[batchid].copyfrom(ts2);
            
        }
    }

    std::map<std::string, Tensor> new_state(size_t max_batch_size = 1){
        std::map<std::string, Tensor> state;
        for (size_t i = 0; i < layers; i++)
        {
            auto wkv = blocks[i].att.state[0];
            auto ts1 = blocks[i].att.timeshift.state[0];
            auto ts2 = blocks[i].ffn.timeshift.state[0];                      
          
            state["blocks." + std::to_string(i) + ".att.state"] = Tensor(wkv.shape, wkv.dtype, wkv.device, wkv.device_id);
            state["blocks." + std::to_string(i) + ".att.timeshift.state"] = Tensor(ts1.shape, ts1.dtype, ts1.device, ts1.device_id);
            state["blocks." + std::to_string(i) + ".ffn.timeshift.state"] = Tensor(ts2.shape, ts2.dtype, ts2.device, ts2.device_id);
            
        }
        return state;
    }

    void cuda(int device = 0){

        emb1.cuda();
        ln0.cuda();
        ln_out.cuda();
        output.cuda();

        for (size_t i = 0; i < layers; i++)
        {
            blocks[i].ln1.cuda();
            blocks[i].ln2.cuda();
            blocks[i].att.gate.cuda();
            blocks[i].att.key.cuda();
            blocks[i].att.value.cuda();
            blocks[i].att.receptance.cuda();
            blocks[i].att.output.cuda();
            blocks[i].att.ln_x.cuda();
            blocks[i].att.timeshift.cuda();
            blocks[i].att.time_decay = blocks[i].att.time_decay.cuda();
            blocks[i].att.time_faaaa = blocks[i].att.time_faaaa.cuda();
            blocks[i].att.time_mix_g = blocks[i].att.time_mix_g.cuda();
            blocks[i].att.time_mix_k = blocks[i].att.time_mix_k.cuda();
            blocks[i].att.time_mix_r = blocks[i].att.time_mix_r.cuda();
            blocks[i].att.time_mix_v = blocks[i].att.time_mix_v.cuda();
            blocks[i].att.state = blocks[i].att.state.cuda();

            blocks[i].ffn.key.cuda();
            blocks[i].ffn.value.cuda();
            blocks[i].ffn.receptance.cuda();
            blocks[i].ffn.timeshift.cuda();
            blocks[i].ffn.time_mix_k = blocks[i].ffn.time_mix_k.cuda();
            blocks[i].ffn.time_mix_r = blocks[i].ffn.time_mix_r.cuda();
  
        }
        
    }
};
