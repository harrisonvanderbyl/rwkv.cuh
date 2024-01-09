#include <iostream>
#include <string>
#include <fstream>
#include "rwkv.h"
// #include "sampler/sample.hpp"
#include "tokenizer/tokenizer.hpp"
#include "chrono"

int main( int argc, char** argv ){

    std::cout << "Hello World" << std::endl;
    std::string path = "../../../../CUDAMAX/1b5model.safetensors";

    
    RWKVTokenizer worldTokenizer("rwkv_vocab_v20230424.txt");
    
    auto tokens = worldTokenizer.encode("\n\nUser: please create a long harry potter fanfiction. \n\nAssistant:");

    
    
    std::cout << worldTokenizer.decode(tokens) << std::endl;
    std::cout << "Loading model" << std::endl;

    // allocating ram for 50 tokens simultaneously
    // used for allocations of static memory usage

    RWKV model(path);

    if (argc > 1)
    {
        model.cuda();
       
    }

    

    std::cout << "Model loaded" << std::endl;

    std::cout << "Layers:" << model.layers << std::endl;
    std::cout << "Embed:" << model.emb1.weight.shape[0] << "," << model.emb1.weight.shape[1] << std::endl;
    std::cout << "Head:" << model.blocks[0].att.ln_x.heads << std::endl;
    // model.cuda();

    auto logits = model({tokens});

    std::cout << logits << std::endl;

    // model.set_state(model.new_state());


    // model.toVulkan();

    // logits = model({tokens});

    // std::cout << logits << std::endl;

    // model.set_state(model.new_state());

    

    // std::cout << "Model sent to vulkan" << std::endl;

    // logits = model({tokens});

    // std::cout << logits << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    size_t tokenstogen = 100;
    std::vector<size_t> generated;
    for (int i = 0; i < tokenstogen; i++)
    {
        // std::cout << "Generating token " << i << std::endl;
        
        float* logs = (float*)(logits[0][logits.shape[1]-1]).cpu().float32().data;
        size_t sample = 0;
        float max = -99999;
        for (int j = 0; j < logits.shape[2]; j++)
        {
            if (float(logs[j]) > max)
            {
                max = float(logs[j]);
                sample = j;
            }
        }

        generated.push_back(sample);

        std::cout.flush();
        std::cout << worldTokenizer.decode({sample});

        logits = model({{sample}});
        // std::cout << logits << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::endl;

    std::cout << "Generated " << tokenstogen << " tokens in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << "tokens per second: " << (tokenstogen / (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0)) << std::endl;

    
    return 0;
}