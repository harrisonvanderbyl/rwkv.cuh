#ifndef SIGMOIDMUL_HPP
#define SIGMOIDMUL_HPP

#include "tensor/tensor.h"


void sigmoidmul_cpu_kernel(void* input, void* other, void* residual, void* output, int size, TENSORTYPE dtype);
void sigmoidmul_cuda_kernel(void* input, void* other, void* residual, void* output, int size, TENSORTYPE dtype);

#ifndef __CUDACC__
void sigmoidmul_cuda_kernel(void* input, void* other, void* residual, void* output, int size, TENSORTYPE dtype){
    printf("not built with CUDA\n");
    exit(0);
}
#endif


Tensor Tensor::sigmoidmul(Tensor& other, Tensor& residual){
    Tensor output = Tensor(this->shape, this->dtype, this->device, this->device_id);
    
    int size = this->get_element_count();
    if (this->device == DEVICE::CPU){
        sigmoidmul_cpu_kernel(this->data, other.data, residual.data, output.data, size, dtype);
    }
    else{
        sigmoidmul_cuda_kernel(this->data, other.data, output.data, residual.data, size, dtype);
    }
    return output;
}







#endif  // SIGMOIDMUL_HPP