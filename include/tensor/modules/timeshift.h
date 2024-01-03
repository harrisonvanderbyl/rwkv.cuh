#ifndef TIMESHIFT_HPP
#define TIMESHIFT_HPP
#include "tensor/tensor.h"
#include "tensor/safetensors.h"
class TimeShift
{
    public:
        uint32_t shiftamount = 1;

        Tensor state;
        
        ulong max_batch;
        ulong max_seq;
        ulong dims;
        
        TimeShift(){
        }

        TimeShift(const size_t dimsi){
            std::vector<size_t> state_size = {1, 1, dimsi};
            // std::cout << "TimeShift:" << state_size[0] << std::endl;
            this->state = Tensor(state_size);
            
            this->dims = dimsi;
            
        }

        Tensor operator()(Tensor input){
            auto batches = input.shape[0];
            auto seq = input.shape[1];
            Tensor output = Tensor(input.shape, input.dtype, input.device, input.device_id);

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