#ifndef SIGMOIDMUL_HPP
#define SIGMOIDMUL_HPP

#include "tensor/tensor.h"


CPUONLY(sigmoidmul_cpu_kernel(void* input, void* other, void* residual, void* output, size_t size, TENSORTYPE dtype, size_t dims));
CUDAONLY(sigmoidmul_cuda_kernel(void* input, void* other, void* residual, void* output, size_t size, TENSORTYPE dtype))

inline Tensor& Tensor::sigmoidmul(Tensor& other, Tensor& residual, Tensor& output){
    
    size_t size = this->get_element_count();
    if (this->device == DEVICE::CPU){
        sigmoidmul_cpu_kernel(this->data, other.data, residual.data, output.data, size, dtype, this->shape[2]);
    }
    else
    {
        sigmoidmul_cuda_kernel(this->data, other.data, residual.data, output.data, size, dtype);
    }
    return output;
}







#endif  // SIGMOIDMUL_HPP