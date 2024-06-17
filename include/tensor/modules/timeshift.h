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
        Tensor time_mix_x;
        Linear time_maa_w1;
        Linear time_maa_w2;
        bool has_time_mix_x = false;
        
        TimeShift(){
        }

        TimeShift(safetensors& model, std::string prefix, size_t batch_size = 1){

            this->time_mix = model[prefix];

            if(model.contains(prefix+"_x")){
                this->time_mix_x = model[prefix+"_x"];
                has_time_mix_x = true;
                this->time_maa_w1 = Linear(model,prefix+"_w1");
                this->time_maa_w2 = Linear(model,prefix+"_w2");
            }

            std::vector<size_t> state_size = {batch_size, 1, time_mix.shape[1]};
            // std::cout << "TimeShift:" << state_size[0] << std::endl;
            this->state = Tensor(state_size);

            
            this->dims = time_mix.shape[1];

            
        }

        Tensor operator()(Tensor input){

            if (buffer.data == nullptr || buffer.shape[1] * buffer.shape[2] < input.shape[0] * input.shape[1] || buffer.dtype != input.dtype || buffer.device != input.device){
                buffer = Tensor({time_mix.shape[0],input.shape[0],input.shape[1], input.shape[2]}, input.dtype, input.device);
            }

            auto batches = input.shape[0];
            auto seq = input.shape[1];
            Tensor output = buffer.cloneWithFalseReshape({time_mix.shape[0],batches, seq, this->dims});

            if(has_time_mix_x){
                auto threadpool = get_threadpool();
                output.empty();
                auto outputa = output[0].cloneWithFalseReshape({1,output.shape[1],output.shape[2],output.shape[3]});
                threadpool->debug(outputa,"outputa");
                this->time_mix_x.shift(input, outputa, this->state, this->dims, false);
                threadpool->debug(outputa,"ptmx");
                threadpool->sync();
                auto xx = this->time_maa_w1(outputa[0]);
                threadpool->sync();
                threadpool->debug(xx,"post_w1");
                xx.tanh();
                threadpool->debug(xx,"posttanh");
                threadpool->sync();
                auto outputb = this->time_maa_w2(xx);
                threadpool->debug(outputb,"posttmaa2");
                this->time_mix.shift(input,outputb,this->state, this->dims, true);
                
                threadpool->debug(outputb,"posttimemaab");
                return outputb;

            }
            else{
                output.empty();
            }

            this->time_mix.shift(input, output, this->state, this->dims, true);
            
            return output;
        }


        void cuda(){
            this->state = this->state.cuda();
            this->buffer = this->buffer.cuda();
            this->time_mix = this->time_mix.cuda();
        }

};

#endif