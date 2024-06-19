
#ifndef tanh_HPP
#define tanh_HPP

#include "tensor/tensor.h"


CPUONLY(tanh_cpu_kernel(void* input, size_t size, TENSORTYPE dtype));
CUDAONLY(tanh_cuda_kernel(void* input, size_t size, TENSORTYPE dtype))

inline Tensor Tensor::tanh(){
    
    size_t size = this->get_element_count();
    if(size == 0){
        return *this;
    }
    if (this->device == DEVICE::CPU){
        tanh_cpu_kernel(this->data, size, dtype);
    }
    else
    {
        tanh_cuda_kernel(this->data, size, dtype);
    }
    return *this;
}







#endif  // tanh_HPP