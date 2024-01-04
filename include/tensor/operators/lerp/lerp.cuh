#ifndef __LERP_CU__
#define __LERP_CU__

template <typename T>
__global__ void lerp_kernel(T *w, T *a, T *b, T* c, size_t size, size_t loopsize) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float weight = w[idx%loopsize];
        c[idx] = float(b[idx]) * weight + float(a[idx]) * (1 - weight);
    }
}

void lerp_cuda_kernel(void* w, void* A, void* B, void* output, size_t size, size_t loopsize, TENSORTYPE dtype)
{
        size_t block_size = 512;
        size_t num_blocks = (size + block_size - 1) / block_size;
        if (dtype == TENSORTYPE::kFLOAT_32){
            lerp_kernel<<<num_blocks, block_size>>>((float*)w, (float*)A, (float*)B, (float*)output, size, loopsize);
        }
        else if (dtype == TENSORTYPE::kBFLOAT_16){
            lerp_kernel<<<num_blocks, block_size>>>((bfloat16*)w, (bfloat16*)A, (bfloat16*)B, (bfloat16*)output, size, loopsize);
        }
        else {
            throw std::runtime_error("Not implemented for this dtype");
        }
}

#endif  // __LERP_CU__