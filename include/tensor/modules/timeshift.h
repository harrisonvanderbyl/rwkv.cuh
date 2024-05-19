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

            this->time_mix.shift(input, output, this->state, this->dims);
            
            return output;
        }


        void cuda(){
            this->state = this->state.cuda();
            this->buffer = this->buffer.cuda();
            this->time_mix = this->time_mix.cuda();
        }

};

#endif