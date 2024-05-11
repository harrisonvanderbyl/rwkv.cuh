#include "./threading.h"

#include <iostream>
#include <immintrin.h>

void test_threading(){
    std::cout << "Testing threading" << std::endl;
    ThreadPool* pool = new ThreadPool(8);

    float mytest[16*16] = {0};

    

    for (size_t i = 0; i < 16; i++){
        pool->add_job([i, &mytest](){
            // mytest[i%16] += i;
            __m512 a = _mm512_loadu_ps(&mytest[i*16]);
            __m512 b = _mm512_set1_ps(i);
            __m512 c = _mm512_add_ps(a, b);
            _mm512_storeu_ps(&mytest[i*16], c);

        }, i);
    }
    auto await = pool->sync();

    

    pool->start();

    


    // print the previous array
    
    // for (size_t i = 0; i < 16; i++){
    //     std::cout << mytest[i] << std::endl;
    // }

    await();
    
    for (size_t i = 0; i < 16; i++){
        std::cout << mytest[i*16] << std::endl;
    }

    // for (size_t i = 0; i < 16; i++){
    //     pool->add_job([i, &mytest](){
    //         std::cout << mytest[i*16] << std::endl;
    //     }, i);
    // }

    pool->sync()();

    
    std::cout << "Finished testing threading" << std::endl;
}


int main(){
    test_threading();
    return 0;
}