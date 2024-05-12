#ifndef TIMESHIFT_HPP
#define TIMESHIFT_HPP
#include "tensor/tensor.h"
#include "tensor/safetensors.h"

#include "tensor/operators/threading/threading.h"
class TimeShift
{
    public:
        uint32_t shiftamount = 1;

        Tensor state;
        
        size_t max_batch;
        size_t max_seq;
        size_t dims;
        Tensor buffer;
        Tensor time_mix;
        
        TimeShift(){
        }

        TimeShift(Tensor time_mix){
            std::vector<size_t> state_size = {16, 1, time_mix.shape[1]};
            // std::cout << "TimeShift:" << state_size[0] << std::endl;
            this->state = Tensor(state_size);

            this->time_mix = time_mix;
            
            this->dims = time_mix.shape[1];

            
        }

        Tensor operator()(Tensor input){

            if (buffer.data == nullptr || buffer.shape[1] * buffer.shape[2] < input.shape[0] * input.shape[1] || buffer.dtype != input.dtype || buffer.device != input.device){
                buffer = Tensor({time_mix.shape[0],input.shape[0],input.shape[1], input.shape[2]}, input.dtype, input.device);
            }

            auto batches = input.shape[0];
            auto seq = input.shape[1];
            Tensor output = buffer.cloneWithFalseReshape({time_mix.shape[0],batches, seq, this->dims});

            auto threadpool = get_threadpool();

            // threadpool->sync();

            auto inputdata = flp(input.data);
            auto outputdata = flp(output.data);
            auto mixdata = flp(time_mix.data);
            auto datatype = (input.dtype);
            auto statedata = flp(this->state.data);
            auto dims = this->dims;

            auto outputchannels = output.shape[0];

            if (datatype != kFLOAT_32){
                throw std::runtime_error("Only float32 is supported for now");
            }
            auto headsize = dims/threadpool->heads;

            for (size_t i = 0; i < threadpool->heads; i++){


                threadpool->add_job([ i, inputdata, outputdata, mixdata, dims, outputchannels, seq, batches,headsize, statedata](){
                    auto btc = batches*seq*dims;
                    auto startofmix = mixdata ;
                    for (size_t j = 0; j < outputchannels; j+=1){
                        auto startofmix = mixdata + j*dims + i*headsize;
                        for (size_t k = 0; k < batches; k+=1){
                            for (size_t l = 0; l < seq; l+=1){

                               auto startofinput = inputdata + k*seq*dims + l*dims + i*headsize;
                               auto startofoutput = outputdata + j*btc + k*seq*dims + l*dims + i*headsize;
                               auto startofmixin = l != 0 ? inputdata + k*seq*dims + (l-1)*dims + i*headsize : statedata + k*dims + i*headsize;
                               lerp_cpu_kernel(startofmix, startofmixin,startofinput,  startofoutput, headsize, headsize, kFLOAT_32);
                               if (l == seq-1 & j == outputchannels-1){
                                    // copy the last output to the state
                                    memcpy(statedata + k*dims + i*headsize, startofinput, headsize*sizeof(float));
                               }
                            }
                        }
                    }


                },i);

            }
            
            threadpool->sync();
            return output;
        }


        void cuda(){
            this->state = this->state.cuda();
        }

};

#endif