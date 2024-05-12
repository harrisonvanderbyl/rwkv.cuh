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

#include "tensor/operators/threading/threading.h"
RWKVTokenizer worldTokenizer("rwkv_vocab_v20230424.txt");

void run(RWKV* model,Tensor logitsin)
{
    // std::cout << "Generating token " << i << std::endl;

    auto pool = get_threadpool();
    auto logs = (logitsin[0][logitsin.shape[1] - 1]);
    size_t sample = dart((float *)logs.data, 0.5, 0.75);
    std::string output = "";
    if (sample == 0)
    {
        output = "user";
        std::cout << "Error:0";
    }
    else
    {
        output = worldTokenizer.decode({sample});
    }

    // lock cout

    auto vnn = output;
    if (output == "user")
    {
        vnn += ": ";
    }

    std::cout << vnn;
    // flush cout
    std::cout << std::flush;

    if (output == "user")
    {
        std::string input = "";
        std::getline(std::cin, input);
        std::cout << "\n";
        auto logits = model->operator()({worldTokenizer.encode("user: " + input + "\n\n")});
        pool->add_job(
            [logits, model]()
            {
                run(model,logits);
            },
            0);
    }
    else
    {

        auto logits = model->operator()({{sample}});
        pool->add_job(
            [logits, model]()
            {
                run(model,logits);
            },
            0);
    }
};

int main(int argc, char **argv)
{

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

    std::string instruction = "user: ";

    std::cout << instruction;

    std::string input = "";
    std::getline(std::cin, input);

    std::cout << "\n";

    auto tokens = worldTokenizer.encode(instruction + input + "\n\n" + "assistant:");

    // model.cuda();
    auto logitsstart = model({tokens});

    auto pool = get_threadpool();

    pool->add_job(
        [logitsstart,&model]()
        {
            run(&model, logitsstart);
        },
        0);

    while (1)
    {
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }
}