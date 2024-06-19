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
        Tensor time_mix_x;
        Linear time_maa_w1;
        Linear time_maa_w2;
        bool has_time_mix_x = false;
        
        TimeShift(){
        }

        TimeShift(safetensors& model, std::string prefix, size_t batch_size = 1){


            if(model.contains(prefix+"_x")){
                this->time_mix_x = model[prefix+"_x"];
                has_time_mix_x = true;
                this->time_maa_w1 = Linear(model,prefix+"_w1", TANH);
                this->time_maa_w2 = Linear(model,prefix+"_w2");
            }

            std::vector<size_t> state_size = {batch_size, 1, time_maa_w2.bias.shape[1]};
            // std::cout << "TimeShift:" << state_size[0] << std::endl;
            this->state = Tensor(state_size);

            
            this->dims = time_maa_w2.bias.shape[1];

            
        }

        Tensor operator()(Tensor input){

            if (buffer.data == nullptr || buffer.shape[1] * buffer.shape[2] < input.shape[0] * input.shape[1] || buffer.dtype != input.dtype || buffer.device != input.device){
                buffer = *new Tensor({time_maa_w2.bias.shape[0],input.shape[0],input.shape[1], input.shape[2]}, input.dtype, input.device);
            }

            auto batches = input.shape[0];
            auto seq = input.shape[1];
            Tensor output = buffer.cloneWithFalseReshape({time_maa_w2.bias.shape[0],batches, seq, this->dims});
            

            auto threadpool = get_threadpool();
            this->time_mix_x.shift(input, output[0], this->state, this->dims, false);
            check_for_errors();
            threadpool->debug(output[0], "ptmx");
            auto xx = this->time_maa_w1(output[0]);

            check_for_errors();
            // threadpool->debug(xx, "ptmaa_w1");
            threadpool->sync();
            // xx.tanh();
            auto outputb = this->time_maa_w2(xx);

            check_for_errors();
             threadpool->debug(outputb, "ptmaa_w2");
            threadpool->sync();
            // threadpool->sync();
            
            return outputb.shift(input,output,this->state, this->dims, true);
        }


        void cuda(){
            this->state = this->state.cuda();
            this->buffer = this->buffer.cuda();

            this->time_mix_x = this->time_mix_x.cuda();
            this->time_maa_w1.cuda();
            this->time_maa_w2.cuda();

        }

};

#endif