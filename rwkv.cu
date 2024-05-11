#include <iostream>
#include <string>
#include <fstream>
#include "rwkv.h"
#include "chrono"
// #include "sampler/sample.hpp"
#include "tokenizer/tokenizer.hpp"
#include "thread"
#include "atomic"
#include "sampler/sample.h"





int main( int argc, char** argv ){

   

    std::string path = "./model.safetensors";
    if (argc > 1)
    {
        path = argv[1];
    }

    size_t threads = 8;
    if (argc > 2)
    {
        threads = std::stoi(argv[2]);
    }

    RWKV model(path, threads);

    RWKVTokenizer worldTokenizer("rwkv_vocab_v20230424.txt");
    
    std::string instruction = "\n\nsystem: Your role is assist the user in any way they ask \n\nuser: ";
    
    
    std::cout << instruction;

    std::string input = "";
    std::getline(std::cin, input );

    std::cout << "\n";

    auto tokens = worldTokenizer.encode(instruction + input+ "\n\n" + "assistant:");

    

    
    

    // model.cuda();
    auto logits = model({tokens});

    auto pool = get_threadpool();
  
    const std::function<void(Tensor& logits)> run = [&](Tensor& logits){
        // std::cout << "Generating token " << i << std::endl;
        
        auto logs =(logits[0][logits.shape[1]-1]).cpu().float32();
        size_t sample = dart((float*)logs.data, 1.0, 0.4);
        std::string output = "";
        if (sample == 0){
            output = "\n\n";
        }
        else{
            output = worldTokenizer.decode({sample});
        }

        

        // lock cout

            auto vnn = output;
            if (output == "\n\n"){
                vnn += "User: ";
            }
            
            std::cout << vnn;
            // flush cout
            std::cout << std::flush;
            

            if (output == "\n\n"){
                std::string input = "";
                std::getline(std::cin, input);
                std::cout << "\n";
                logits = model({worldTokenizer.encode("\n\nuser: "+input + "\n\nassistant:")});
                
            }else{

                logits = model({{sample}});

            }

            pool->add_job(
                [&](){
                    run(logits);
                },
                0
            );
    };

    pool->add_job(
                [&](){
                    run(logits);
                },
                0
            );

    while(1){
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }
}