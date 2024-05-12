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

    RWKV model(path, threads, true);
    auto pool = get_threadpool();

    // model.cuda();
    auto logits = model({{0}});

    pool->debug(model.blocks[0].attshift.state[0], "attshift state");
    pool->debug(model.blocks[0].ffnshift.state[0], "ffnshift state");
    pool->debug(model.blocks[0].att.state[0], "att state");


    pool->start();

    pool->sync(true);

    pool->stop();

}