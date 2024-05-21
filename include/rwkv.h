#include <iostream>
#include <string>
#include <fstream>
#include "tensor/safetensors.h"
#include "tensor/modules/embedding.h"
#include "tensor/modules/layernorm.h"
#include "tensor/modules/linear.h"
#include "tensor/modules/block.h"

#include "tensor/operators/threading/threading.h"
#ifndef RWKV_H
#define RWKV_H
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

    RWKV(std::string path, size_t threadsNum = 0, bool debug = false)
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
                if (std::string(key).find("ln1.weight") != std::string::npos)
                {
                    layers++;
                }
               
            }
        }
        
        auto pool = get_threadpool(threadsNum, debug);
        pool->start();

        // std::cout << "layers:" << layers << std::endl;

        auto t1o = model["emb.weight"];
        this->emb1 = Embedding(t1o);
        this->ln0 = LayerNorm(model["blocks.0.ln0.weight"], model["blocks.0.ln0.bias"]);
        this->ln_out = LayerNorm(model["ln_out.weight"], model["ln_out.bias"]);
        this->output = Linear(model, "head");
        for (size_t i = 0; i < layers; i++)
        {
            // std::cout << "Block" << i << std::endl;
            blocks.push_back(Block(model, i));
        }

    }

    Tensor operator()(std::vector<std::vector<size_t>> input)
    {
        auto rx = emb1(input);
        auto x = ln0(rx);

        for (size_t i = 0; i < layers; i++)
        {
            blocks[i](x);
        }
        auto xm = ln_out(x);
        check_for_errors();
        auto pool = get_threadpool(); 
        pool->sync();
        // xm = xm.cpu();
        auto out = output(xm);

        check_for_errors();
        pool->sync();
        return out;

    }

    void get_state(std::map<std::string, Tensor> state, size_t batchid = 0){
       
        
        for (size_t i = 0; i < layers; i++)
        {
            auto wkv = blocks[i].att.state[batchid];
            auto ts1 = blocks[i].attshift.state[batchid];
            auto ts2 = blocks[i].ffnshift.state[batchid];
          
            state["blocks." + std::to_string(i) + ".att.state"].copyfrom(wkv);
            state["blocks." + std::to_string(i) + ".attshift.state"].copyfrom(ts1);
            state["blocks." + std::to_string(i) + ".ffnshift.state"].copyfrom(ts2);
            
        }
    }

    void zero_state(size_t batchid = 0){
        for (size_t i = 0; i < layers; i++)
        {
            blocks[i].att.state[batchid].empty();
            blocks[i].attshift.state[batchid].empty();
            blocks[i].ffnshift.state[batchid].empty();
        }
    }

    void set_state(std::map<std::string, Tensor> state, size_t batchid = 0){
        for (size_t i = 0; i < layers; i++)
        {
            auto wkv = state["blocks." + std::to_string(i) + ".att.state"];
            auto ts1 = state["blocks." + std::to_string(i) + ".attshift.state"];
            auto ts2 = state["blocks." + std::to_string(i) + ".ffnshift.state"];

            // std::cout << "wkv:" << wkv.shape[0] << " : " << wkv.shape[1] << std::endl;
            // std::cout << "ts1:" << ts1.shape[0] << " : " << ts1.shape[1] << std::endl;
            // std::cout << "ts2:" << ts2.shape[0] << " : " << ts2.shape[1] << std::endl;
            blocks[i].att.state[batchid].copyfrom(wkv);
            blocks[i].attshift.state[batchid].copyfrom(ts1);
            blocks[i].ffnshift.state[batchid].copyfrom(ts2);
            
        }
    }

    void expand_state(size_t batchsize){
         for (size_t i = 0; i < layers; i++)
        {
            auto newattshape = blocks[i].att.state.shape;
            size_t oldbatch = newattshape[0];
            newattshape[0] = batchsize;
            auto newattstate = new Tensor(newattshape);
            for (size_t bb = 0; bb < oldbatch; bb++)
            {
                newattstate->copyfrom(blocks[i].att.state[bb]);
            }
            blocks[i].att.state.data = newattstate->data;
            

            
            // blocks[i].attshift.state[batchid].copyfrom(ts1);
            auto newattshiftshape = blocks[i].attshift.state.shape;
            newattshiftshape[0] = batchsize;
            auto newattshiftstate = new Tensor(newattshiftshape);
            for (size_t bb = 0; bb < oldbatch; bb++)
            {
                newattshiftstate->copyfrom(blocks[i].attshift.state[bb]);
            }
            blocks[i].attshift.state.data = newattshiftstate->data;

            auto newffnshiftshape = blocks[i].ffnshift.state.shape;
            newffnshiftshape[0] = batchsize;
            auto newffnshiftstate = new Tensor(newffnshiftshape);
            for (size_t bb = 0; bb < oldbatch; bb++)
            {
                newffnshiftstate->copyfrom(blocks[i].ffnshift.state[bb]);
            }
            blocks[i].ffnshift.state.data = newffnshiftstate->data;
            // blocks[i].ffnshift.state[batchid].copyfrom(ts2);
            
        }
    }

    std::map<std::string, Tensor> new_state(size_t max_batch_size = 1){
        std::map<std::string, Tensor> state;
        for (size_t i = 0; i < layers; i++)
        {
            auto wkv = blocks[i].att.state[0];
            auto ts1 = blocks[i].attshift.state[0];
            auto ts2 = blocks[i].ffnshift.state[0];                      
          
            state["blocks." + std::to_string(i) + ".att.state"] = Tensor(wkv.shape, wkv.dtype, wkv.device, wkv.device_id);
            state["blocks." + std::to_string(i) + ".attshift.state"] = Tensor(ts1.shape, ts1.dtype, ts1.device, ts1.device_id);
            state["blocks." + std::to_string(i) + ".ffnshift.state"] = Tensor(ts2.shape, ts2.dtype, ts2.device, ts2.device_id);
            
        }
        return state;
    }

    void cuda(int device = 0){
        std::cout << "Cudaing" << std::endl;
        emb1.cuda();
        std::cout << "Cudaed emb1" << std::endl;
        ln0.cuda();
        std::cout << "Cudaed ln0" << std::endl;
        ln_out.cuda();
        std::cout << "Cudaed ln_out" << std::endl;
        output.cuda();
        std::cout << "Cudaed output" << std::endl;

        for (size_t i = 0; i < layers; i++)
        {
            std::cout << "Cudaing block" << i << std::endl;
            blocks[i].ln1.cuda();
            blocks[i].ln2.cuda();
            blocks[i].att.gate.cuda();
            blocks[i].att.key.cuda();
            blocks[i].att.value.cuda();
            blocks[i].att.receptance.cuda();
            blocks[i].att.output.cuda();
            blocks[i].att.ln_x.cuda();
            blocks[i].attshift.cuda();
            blocks[i].ffnshift.cuda();
            blocks[i].att.time_decay = blocks[i].att.time_decay.cuda();
            blocks[i].att.time_faaaa = blocks[i].att.time_faaaa.cuda();
            blocks[i].att.state = blocks[i].att.state.cuda();

            blocks[i].ffn.key.cuda();
            blocks[i].ffn.value.cuda();
            blocks[i].ffn.receptance.cuda();
  
        }
        
    }
};



static ThreadPool* threadpool = nullptr;
ThreadPool* __attribute__((weak)) get_threadpool(size_t threadsNum, bool debug){
    if (threadpool == nullptr)
    {
        if (threadsNum == 0)
        {
            std::cout << "Initial threadpool not specified, using hardware concurrency" << std::endl;
            threadsNum = std::thread::hardware_concurrency();
            std::cout << "Hardware concurrency: " << threadsNum << std::endl;
            // restrict threadsnum to power of 2
            threadsNum = 1 << (int)log2(threadsNum);
        }
        threadpool = new ThreadPool(threadsNum, debug);
    }
    return threadpool;
}

#endif