#ifndef __LERP_CU__
#define __LERP_CU__

template <typename T>
__global__ void lerp_kernel(T* inputdata, T* outputdata, T* mixdata, size_t dims, size_t outputchannels, size_t seq, size_t batches, size_t headsize, T* statedata)
{
    auto k = blockIdx.x;
    auto i = threadIdx.x;
       
    auto btc = batches*seq*dims;
    for (size_t j = 0; j < outputchannels; j+=1){
        auto startofmix = mixdata + j*dims + i*headsize;
        
            for (size_t l = 0; l < seq; l+=1){

                auto startofinput = inputdata + k*seq*dims + l*dims + i*headsize;
                auto startofoutput = outputdata + j*btc + k*seq*dims + l*dims + i*headsize;
                auto startofmixin = l != 0 ? inputdata + k*seq*dims + (l-1)*dims + i*headsize : statedata + k*dims + i*headsize;
                for (size_t m = 0; m < headsize; m+=1){
                    startofoutput[m] = startofinput[m] * startofmix[m] + startofmixin[m] * (T(1)-startofmix[m]);
                }
                if (l == seq-1 & j == outputchannels-1){
                    // copy the last output to the state
                    for (size_t m = 0; m < headsize; m+=1){
                        statedata[k*dims + i*headsize + m] = startofinput[m];
                    }
                    // memcpy(statedata + k*dims + i*headsize, startofinput, headsize*sizeof(float));
                }
            }
        
    }
}

void lerp_cuda_kernel(void* inputdata, void* outputdata, void* mixdata, size_t dims, size_t outputchannels, size_t seq, size_t batches, void* statedata, TENSORTYPE dtype)
{
        
        size_t blocks = batches;

        size_t headsize = 64;

        size_t threads = dims/headsize;
        


        if (dtype == TENSORTYPE::kFLOAT_32){
            lerp_kernel<<<blocks, threads>>>( (float*)inputdata, (float*)outputdata, (float*)mixdata, dims, outputchannels, seq,  batches, headsize, (float*)statedata);
        }
        else if (dtype == TENSORTYPE::kBFLOAT_16){
            lerp_kernel<<<blocks, threads>>>((bfloat16*)inputdata, (bfloat16*)outputdata, (bfloat16*)mixdata, dims, outputchannels, seq,  batches, headsize, (bfloat16*)statedata);
        }
        else {
            throw std::runtime_error("Not implemented for this dtype");
        }
}

#endif  // __LERP_CU__