#ifndef HVMLAVX512MAT8_CPP
#define HVMLAVX512MAT8_CPP
#include "tensor/tensor.h"
#include <iostream>



CPUONLY(matmul8_cpu_kernal(uint8_t* A, void* B, void* C, void* Ao, void* Ar, size_t BBT, size_t INSHAPE, size_t OUTSHAPE, MMACTFUNC func));
CUDAONLY(matmul8_cuda_kernal(uint8_t* A, void* B, void* C, void* Ao, void* Ar, size_t BBT, size_t INSHAPE, size_t OUTSHAPE, MMACTFUNC func))

CPUONLY(matmul_cpu_kernal(void* A, void* B, void* C, size_t BBT, size_t INSHAPE, size_t OUTSHAPE, TENSORTYPE dtype, size_t bmmshape, MMACTFUNC func));
CUDAONLY(matmul_cuda_kernal(void* A, void* B, void* C, size_t BBT, size_t INSHAPE, size_t OUTSHAPE,TENSORTYPE dtype, size_t bmmshape, MMACTFUNC func))
        
CPUONLY(wkv5_cpu_kernel(void* kk, void* vv, void* ww, void* uu, void* rr, void* ss, void* out, size_t T, size_t B, size_t C, size_t H, TENSORTYPE dtype))
CUDAONLY(wkv5_cuda_kernel(void* kk, void* vv, void* ww, void* uu, void* rr, void* ss, void* out, size_t T, size_t B, size_t C, size_t H, TENSORTYPE dtype))

inline Tensor Tensor::matmul(Tensor &Art, Tensor &Aot,
                      Tensor &Bt, Tensor Ct, MMACTFUNC func)
{
    assert(this->shape.size() < 3);
    // Pointers to the data
    if (Bt.dtype != TENSORTYPE::kFLOAT_32)
    {
        printf("only float32 embs allowed for cpu uint8 matmul\n");
        exit(1);
    }
    const uint8_t *A = (uint8_t *)this->data;
    const auto Ar = Art.data;
    const auto Ao = Aot.data;
    const auto B = Bt.data;
    if (Ct.data == nullptr)
    {
        Ct =* new Tensor({Bt.shape[0], Bt.shape[1], this->shape[0]}, Bt.dtype, Bt.device, Bt.device_id);
    }
    const auto C = Ct.data;

    const size_t BB = Bt.shape[0];
    const size_t T = Bt.shape[1];
    const size_t INSHAPE = Bt.shape[2];
    const size_t OUTSHAPE = this->shape[0];

    if (Bt.device == DEVICE::CPU)
    {
        matmul8_cpu_kernal((uint8_t *)A, (void *)B, (void *)C, (void *)Ao, (void *)Ar, BB * T, INSHAPE, OUTSHAPE, func);
    }
    else
    {

        
        matmul8_cuda_kernal((uint8_t *)A, (void *)B, (void *)C, (void *)Ao, (void *)Ar, BB * T, INSHAPE, OUTSHAPE, func);
       
    }

    return Ct;
}

inline Tensor Tensor::matmul(Tensor &Bt, Tensor Ct, MMACTFUNC func)
{
    if(this->data == nullptr || data_size_in_bytes == 0){
        return Ct;
    }
    // Pointers to the data
    if (Bt.dtype != TENSORTYPE::kFLOAT_32 && Bt.dtype != TENSORTYPE::kBFLOAT_16)
    {
        std::cout << "only float32 or bfloat16 embs allowed for matmul" << std::endl;
        exit(1);
    }
    const auto A = this->data;
    const auto B = Bt.data;

    size_t bmmsize = 1;
    size_t OUTSHAPE = this->shape[0];
    size_t INSHAPE = this->shape[1];
    size_t BB = Bt.shape[0];
    size_t T = Bt.shape[1];

    if(this->shape.size() == 3){
        bmmsize = this->shape[0];
        OUTSHAPE = this->shape[1];
        INSHAPE = this->shape[2];
    }

    if (Ct.data == nullptr)
    {
        Ct = Tensor({BB*bmmsize, T, OUTSHAPE}, Bt.dtype, Bt.device, Bt.device_id);
    }

    // std::cout << Ct;
    // std::cout << *this;
    // std::cout << Bt;

    const auto C = Ct.data;

    
    

    if(Bt.device==DEVICE::CPU){
        matmul_cpu_kernal((void *)A, (void *)B, (void *)C, BB * T, INSHAPE, OUTSHAPE, this->dtype, bmmsize, func);
    } 
    else
    {
    
        
        matmul_cuda_kernal((void *)A, (void *)B, (void *)C, BB * T, INSHAPE, OUTSHAPE, this->dtype, bmmsize, func);
        
    }

    return Ct;
}

inline Tensor Tensor::wkv5(Tensor &r, Tensor &k, Tensor &v, Tensor &w, Tensor &u, Tensor &y)
{

    auto rr = r.data;
    auto kk = k.data;
    auto vv = v.data;
    auto ww = w.data;
    auto uu = u.data;
    float *ss = (float *)(this->data);
    auto out = y.data;
    auto rdtype = r.dtype;

    uint32_t B = r.shape[0];
    uint32_t T = r.shape[1];
    uint32_t C = r.shape[2];
    uint32_t H = this->shape[1];

    if (device == DEVICE::CPU)
    {
        wkv5_cpu_kernel(kk, vv, ww, uu, rr, ss, out, T, B, C, H, rdtype);
    }
    else
    {

        
        wkv5_cuda_kernel(kk, vv, ww, uu, rr, ss, out, T, B, C, H, rdtype);
        
    }

    return y;
}

#endif // HVMLAVX512MAT8_CPP