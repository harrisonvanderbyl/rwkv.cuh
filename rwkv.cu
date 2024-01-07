#include <iostream>
#include <string>
#include <fstream>
#include "rwkv.h"
#include "chrono"
// #include "sampler/sample.hpp"
#include "tokenizer/tokenizer.hpp"
#include "thread"
#include "atomic"

struct wordo {
    std::string word;
    size_t count;
    size_t timetoflush;
};

wordo wchain1 = wordo{"", 0,1};
wordo wchain2 = wordo{"", 1,1};

std::atomic<const wordo*> coutbuffer;


void outputthreadfunc(){
    while(1){
        
        // switch and reset cout

        const wordo* cout = coutbuffer.exchange(&wchain1);
        std::string unflushed = cout->word;
        // get cout
        // std::getline(coutbuffer, unflushed);
        // reset cout
        auto timetoflush = cout->timetoflush;
        if (unflushed.length() == 0){
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        auto timeperchar = timetoflush / (unflushed.length()+1);
        for (int i = 0; i < unflushed.length(); i++)
        {
            
            std::cout << unflushed[i];
            // // flush cout
            std::cout.flush();
            //wait for 0.01 seconds
            std::this_thread::sleep_for(std::chrono::milliseconds(timeperchar));
            
        }
    }
}

int main( int argc, char** argv ){
    coutbuffer.store(&wchain1);

    std::string path = "./model.safetensors";
    if (argc > 1)
    {
        path = argv[1];
    }

    RWKV model(path);

    RWKVTokenizer worldTokenizer("rwkv_vocab_v20230424.txt");
    
    std::string instruction = "\n\nSystem: Your role is assist the user in fulfilling their fantasies by creating a world simulation. \n\nUser: ";
    
    
    std::cout << instruction;

    std::string input = "";
    std::getline(std::cin, input );

    std::cout << "\n";

    auto tokens = worldTokenizer.encode(instruction + input+ "\n\n");

    if (argc > 2)
    {
        model.cuda();
    }

    


    // model.cuda();
    auto logits = model({tokens});

    // create thread to flush cout
    std::thread outputthread(outputthreadfunc);

    auto start = std::chrono::high_resolution_clock::now();
    auto last = start;
    size_t tokenstogen = 10000000;
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

        std::string output = worldTokenizer.decode({sample});

        auto t = std::chrono::high_resolution_clock::now();


        // if output ends with User
        
            
            
        // lock cout
            const wordo* cout = coutbuffer.exchange(&wchain1);

            auto vnn = output;
            if (output == "\n\n"){
                vnn += "User: ";
            }
            
            if (cout->count == wchain2.count){
                wchain2.word += vnn;
                }else{
                wchain2.word = vnn ;
            }
            wchain2.timetoflush = (wchain2.timetoflush + std::chrono::duration_cast<std::chrono::milliseconds>(t - last).count())/2.0;
            
            last = t;

            coutbuffer.store(&wchain2);

            if (output == "\n\n"){
                std::string input = "";
                std::getline(std::cin, input);
                std::cout << "\n";
                last = std::chrono::high_resolution_clock::now();
                logits = model({worldTokenizer.encode("\n\nUser: "+input + "\n\n")});
                
                
            }else{
                logits = model({{sample}});
            }


        
        

        
        // std::cout << logits << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::endl;

    std::cout << "Generated " << tokenstogen << " tokens in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    std::cout << "tokens per second: " << (tokenstogen / (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0)) << std::endl;

    
    return 0;
}