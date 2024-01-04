#ifndef RELUSQUARE_HPP
#define RELUSQUARE_HPP

#include "tensor/tensor.h"

void relusquare_cpu_kernel(void* input, void* output, int size, TENSORTYPE dtype);
void relusquare_cuda_kernel(void* input, void* output, int size, TENSORTYPE dtype);

inline Tensor Tensor::relusquared(){
    Tensor output = Tensor(this->shape, this->dtype, this->device, this->device_id);
    size_t size = this->get_element_count();

    if (device == DEVICE::CPU){
        relusquare_cpu_kernel(this->data, output.data, size, dtype);
    }
    else if (device == DEVICE::CUDA){
    
        relusquare_cuda_kernel(this->data, output.data, size, dtype);
        
    }
    return output;
}


#endif // RELUSQUARE_HPP