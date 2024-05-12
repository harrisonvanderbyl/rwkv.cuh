

// Cpu operators
#include "tensor/operators/sigmoidmul/cpu.h"
#include "tensor/operators/swishmul/cpu.h"
#include "tensor/operators/relusquare/cpu.h"
#include "tensor/operators/normalize/cpu.h"
#include "tensor/operators/lerp/cpu.h"
#include "tensor/operators/matmul/cpu.h"
#include "tensor/operators/matmul/threading.h"
#include "tensor/operators/threading/threading.h"

static ThreadPool* threadpool = nullptr;
ThreadPool* get_threadpool(size_t threadsNum, bool debug){
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
