#ifndef LERP_HPP    
#define LERP_HPP

#include "tensor/tensor.h"
#include "tensor/operators/threading/threading.h"

CUDAONLY(lerp_cuda_kernel(void* inputdata, void* outputdata, void* mixdata, size_t dims, size_t outputchannels, size_t seq, size_t batches, void* statedata, TENSORTYPE dtype, bool initiate_move, size_t mixsize))
CPUONLY(lerp_cpu_kernel(void* w, void* A, void* B, void* output, size_t size, TENSORTYPE dtype))


inline Tensor Tensor::shift(Tensor input, Tensor output, Tensor& state, size_t indims, bool initiate_move){
    auto batches = input.shape[0];
    auto seq = input.shape[1];
    auto mixsize = get_element_count();
    if (this->device == DEVICE::CPU){
        auto threadpool = get_threadpool();

            // threadpool->sync();

        auto inputdata = flp(input.data);
        auto outputdata = flp(output.data);
        auto mixdata = flp(data);
        auto datatype = (input.dtype);
        auto statedata = flp(state.data);
        auto dims = indims;

        auto outputchannels = shape[0];

        if (datatype != kFLOAT_32){
            throw std::runtime_error("Only float32 is supported for now");
        }
        auto headsize = dims/threadpool->heads;

        

        for (size_t i = 0; i < threadpool->heads; i++){


            threadpool->add_job([ i, inputdata, outputdata, mixdata, dims, outputchannels, seq, batches,headsize, statedata, mixsize, initiate_move](){
                auto btc = batches*seq*dims;
                for (size_t j = 0; j < outputchannels; j+=1){
                    for (size_t k = 0; k < batches; k+=1){
                        for (size_t l = 0; l < seq; l+=1){

                            auto startofinput = inputdata + k*seq*dims + l*dims + i*headsize;
               
                            auto startofmix = mixdata + (j*btc + k*seq*dims + l*dims + i*headsize)%mixsize;
                           

                            auto startofoutput = outputdata + j*btc + k*seq*dims + l*dims + i*headsize;
                            auto startofmixin = l != 0 ? inputdata + k*seq*dims + (l-1)*dims + i*headsize : statedata + k*dims + i*headsize;
                            lerp_cpu_kernel(startofmix, startofinput, startofmixin,  startofoutput, headsize, kFLOAT_32);
                            if (l == seq-1 && j == outputchannels-1 && initiate_move){
                                // copy the last output to the state
                                memcpy(statedata + k*dims + i*headsize, startofinput, headsize*sizeof(float));
                            }
                        }
                    }
                }


            },i);

        }
        
        threadpool->sync();
    }
    else
    {
        if(shape[0] == 0){
            return output;
        }
        lerp_cuda_kernel(input.data, output.data, data, indims, shape[0], seq, batches, state.data, dtype, initiate_move, mixsize);
    }  

    
    return output;
}


#endif // LERP_HPP