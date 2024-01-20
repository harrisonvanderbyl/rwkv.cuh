#ifndef tahn_HPP
#define tahn_HPP

#include "tensor/tensor.h"


void tahn_cpu_kernel(void* input, void* output, size_t size, TENSORTYPE dtype);
void tahn_cuda_kernel(void* input, void* output, size_t size, TENSORTYPE dtype);

inline Tensor Tensor::tahn(){
    
    size_t size = this->get_element_count();
    if (this->device == DEVICE::CPU){
        tahn_cpu_kernel(this->data, this->data,  size, dtype);
    }
    else CUDAONLY
    {
        tahn_cuda_kernel(this->data, this->data, size, dtype);
    }
    return *this;
}







#endif  // tahn_HPP