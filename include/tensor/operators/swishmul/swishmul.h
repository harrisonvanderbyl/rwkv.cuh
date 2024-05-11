#ifndef SWISHMUL_HPP
#define SWISHMUL_HPP

#include "tensor/tensor.h"


void swishmul_cpu_kernel(void* input, void* other, void* output, size_t size, TENSORTYPE dtype, size_t dims);
void swishmul_cuda_kernel(void* input, void* other, void* output, size_t size, TENSORTYPE dtype);

inline Tensor Tensor::swishmul(Tensor& other){
    Tensor output = *this;
    
    size_t size = this->get_element_count();
    if (this->device == DEVICE::CPU){
        swishmul_cpu_kernel(this->data, other.data, output.data, size, dtype, this->shape[2]);
    }
    else CUDAONLY
    {
        swishmul_cuda_kernel(this->data, other.data, output.data, size, dtype);
    }
    return output;
}







#endif  // SWISHMUL_HPP