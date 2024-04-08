#ifndef TIMESHIFT_HPP
#define TIMESHIFT_HPP
#include "tensor/tensor.h"
#include "tensor/safetensors.h"
class TimeShift
{
    public:
        uint32_t shiftamount = 1;

        Tensor state;
        
        size_t max_batch;
        size_t max_seq;
        size_t dims;
        Tensor buffer;
        
        TimeShift(){
        }

        TimeShift(const size_t dimsi){
            std::vector<size_t> state_size = {16, 1, dimsi};
            // std::cout << "TimeShift:" << state_size[0] << std::endl;
            this->state = Tensor(state_size);
            
            this->dims = dimsi;
            
        }

        Tensor operator()(Tensor input){

            if (buffer.data == nullptr || buffer.shape[0] * buffer.shape[1] < input.shape[0] * input.shape[1] || buffer.dtype != input.dtype || buffer.device != input.device){
                buffer = Tensor({input.shape[0],input.shape[1], input.shape[2]}, input.dtype, input.device);
            }

            auto batches = input.shape[0];
            auto seq = input.shape[1];
            Tensor output = buffer.cloneWithFalseReshape({batches, seq, this->dims});

            for (size_t i = 0; i < batches; i++){
                output[i][0].copyfrom(this->state[i][0]);
                for (size_t j = 0; j < seq; j++){
                    if (j > 0){
                        output[i][j].copyfrom(input[i][j-1]);
                    }
                    else{
                        this->state[i][0].copyfrom(input[i][seq-1]);
                    }
                }
            }
            return output;
        }


        void cuda(){
            this->state = this->state.cuda();
        }

};

#endif