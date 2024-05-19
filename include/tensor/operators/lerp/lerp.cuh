#ifndef __LERP_CU__
#define __LERP_CU__

__global__ void lerp_kernel(float *inputdata, float *outputdata, float *mixdata, size_t dims, size_t outputchannels, size_t seq, size_t batches, size_t headsize, float *statedata)
{
    auto k = blockIdx.x;
    auto j = blockIdx.y;
    auto i = threadIdx.x;

    auto btc = batches * seq * dims;
    auto startofmix = mixdata + j * dims + i * headsize;

    for (size_t l = 0; l < seq; l += 1)
    {

        auto startofinput = inputdata + k * seq * dims + l * dims + i * headsize;
        auto startofoutput = outputdata + j * btc + k * seq * dims + l * dims + i * headsize;
        auto startofmixin = l != 0 ? inputdata + k * seq * dims + (l - 1) * dims + i * headsize : statedata + k * dims + i * headsize;
        for (size_t m = 0; m < headsize; m += 1)
        {
            startofoutput[m] = startofinput[m] * startofmix[m] + startofmixin[m] * (1.0f - startofmix[m]);
        }
        if (l == seq - 1 & j == outputchannels - 1)
        {
            // copy the last output to the state
            for (size_t m = 0; m < headsize; m += 1)
            {
                statedata[k * dims + i * headsize + m] = startofinput[m];
            }
            // memcpy(statedata + k*dims + i*headsize, startofinput, headsize*sizeof(float));
        }
    }
}

void lerp_cuda_kernel(void *inputdata, void *outputdata, void *mixdata, size_t dims, size_t outputchannels, size_t seq, size_t batches, void *statedata, TENSORTYPE dtype)
{

    auto blocks = dim3(batches, outputchannels, 1);
    size_t threads = 512;

    size_t headsize = dims / threads;

    if (dtype == TENSORTYPE::kFLOAT_32)
    {
        lerp_kernel<<<blocks, threads>>>((float *)inputdata, (float *)outputdata, (float *)mixdata, dims, outputchannels, seq, batches, headsize, (float *)statedata);
    }
    else
    {
        throw std::runtime_error("Not implemented for this dtype");
    }
}

#endif // __LERP_CU__