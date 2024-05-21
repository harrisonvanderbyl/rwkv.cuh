#ifndef RELUSQUARE_HPP
#define RELUSQUARE_HPP

#include "tensor/tensor.h"

static void relusquare_cpu_kernel(void* input, void* output, size_t size, TENSORTYPE dtype, size_t dims);
CUDAONLY(relusquare_cuda_kernel(void* input, void* output, size_t size, TENSORTYPE dtype))

inline Tensor Tensor::relusquared(){
    Tensor output = *this;
    size_t size = this->get_element_count();

    if (device == DEVICE::CPU){
        relusquare_cpu_kernel(this->data, output.data, size, dtype, this->shape[2]);
    }
    else
    {
    
        relusquare_cuda_kernel(this->data, output.data, size, dtype);
        
    }
    return output;
}


#endif // RELUSQUARE_HPP