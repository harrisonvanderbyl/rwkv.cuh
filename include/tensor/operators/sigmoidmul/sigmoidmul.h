#ifndef SIGMOIDMUL_HPP
#define SIGMOIDMUL_HPP

#include "tensor/tensor.h"


void sigmoidmul_cpu_kernel(void* input, void* other, void* residual, void* output, int size, TENSORTYPE dtype);
void sigmoidmul_cuda_kernel(void* input, void* other, void* residual, void* output, size_t size, TENSORTYPE dtype);

inline Tensor Tensor::sigmoidmul(Tensor& other, Tensor& residual){
    Tensor output = *this;
    
    int size = this->get_element_count();
    if (this->device == DEVICE::CPU){
        sigmoidmul_cpu_kernel(this->data, other.data, residual.data, output.data, size, dtype);
    }
    else{
        sigmoidmul_cuda_kernel(this->data, other.data, residual.data, output.data, size, dtype);
    }
    return output;
}







#endif  // SIGMOIDMUL_HPP