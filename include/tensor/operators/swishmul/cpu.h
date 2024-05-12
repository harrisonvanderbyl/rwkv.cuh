#ifndef SWISHMUL_NAIVE_H
#define SWISHMUL_NAIVE_H
#include "tensor/tensor.h"

#include "tensor/intrinsics/intrinsics.h"


#include "tensor/operators/threading/threading.h"
void swishmul_cpu_kernel(void* input, void* other, void* output, size_t size, TENSORTYPE dtype,size_t dims){

    size_t simdwidth = get_simd_width();

    
    auto pool = get_threadpool();

    auto headsize = dims/pool->heads;

    if (dtype == TENSORTYPE::kFLOAT_32){   

        for (size_t t = 0; t < pool->heads; t++){
            pool->add_job([input, other, output, size, dims, simdwidth, t, headsize]
            {
                for (size_t ii = t*headsize; ii < size; ii+=dims){
                    for (size_t i = ii; i < ii + headsize; i+=simdwidth){
                        simd_swishmul(flp(input) + i, flp(other) + i, flp(output) + i);
                    }
                }
            },t);
        }
        

    }
    
    else{
        throw std::runtime_error("swishmul only implemented for float");
    }
}

#endif //SWISHMUL_NAIVE_H