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

    std::string path = "./3b.safetensors";
    if (argc > 1)
    {
        path = argv[1];
    }

    float temp = 1.0;
    float sampmin = 0.6;

    if (argc > 2)
    {
        temp = atof(argv[2]);
    }  
    if (argc > 3)
    {
        sampmin = atof(argv[3]);
    }    

    std::cout << "Temp: " << temp;

    RWKV model(path);
    

    RWKVTokenizer worldTokenizer("rwkv_vocab_v20230424.txt");
    
    std::string instruction = "\n\nSystem: You are an assistant for the user, running in a bash terminal. Use \nRUN\n{command}\nENDRUN\n to run a command for the user.\n\nUser: Please display test.txt\n\nAssistant: Sure! I can do that!\nRUN\ncat test.txt\nENDRUN\n\n\nUser: Thanks!\n\nAssistant: You're welcome!\n\nUser:";
    
    
    std::cout << instruction << std::endl;

    std::string input = "";
    std::getline(std::cin, input );

    std::cout << "\n";

    auto tokens = worldTokenizer.encode(instruction + input+ "\n\nAssistant:");

   
    // model.cuda();
    auto logits = model({tokens});

    std::cout << logits << std::endl;

    // create thread to flush cout
    std::thread outputthread(outputthreadfunc);

    auto last = std::chrono::high_resolution_clock::now();
    std::string lastcommand = "";
    while(1)
    {
        // std::cout << "Generating token " << i << std::endl;
        
        auto logs =(logits[0][logits.shape[1]-1]).cpu().float32();
        size_t sample = typical((float*)logs.data, temp, sampmin);
     
        std::string output;
        if(sample == 0){
            output = "\n\n";
        }else{
          output = worldTokenizer.decode({sample});
        }

        auto t = std::chrono::high_resolution_clock::now();


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

            lastcommand += output;

            wchain2.timetoflush = (wchain2.timetoflush + std::chrono::duration_cast<std::chrono::milliseconds>(t - last).count())/2.0;
            
            last = t;

            coutbuffer.store(&wchain2);

            if (output == "\n\n"){

                // extract command
                std::string command = lastcommand.substr(lastcommand.find("RUN\n")+4);
                while (command.find("\nENDRUN") != std::string::npos){
                    auto commando = command.substr(0, command.find("\nENDRUN"));
                    // run command
                    system(commando.c_str()); 
                    command = command.substr(command.find("\nENDRUN")+7);
                }
                

                std::string input = "";
                std::getline(std::cin, input);
                std::cout << "\n";
                logits = model({worldTokenizer.encode("\n\nUser: "+input + "\n\nAssistant:")});
                last = std::chrono::high_resolution_clock::now();
                lastcommand = "";
                
            }else{
                logits = model({{sample}});
            }
    }
    
    return 0;
}