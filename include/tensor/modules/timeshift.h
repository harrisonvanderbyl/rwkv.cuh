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

// class LoRA{
//     public:
//         Linear wb;
//         Linear wa;

//     LoRA(){}

//     LoRA(safetensors& model, std::string prefix){
//         wa = Linear(model, prefix + "_w1");
//         wb = Linear(model, prefix + "_w2");
//     }
// }

        /*
class LoRA(nn.Module):
    def __init__(self, dim:int, dim_hidden:int, init_value : Tensor|None = None):
        super().__init__()
        init_value = init_value if init_value is not None else torch.zeros(dim)
        self.base = nn.Parameter(init_value)
        self.Wa = nn.Linear(dim, dim_hidden, bias=False)
        self.Wb = nn.Linear(dim_hidden, dim, bias=False)

    def forward(self, x : Tensor): # x (B,T,C)
        # this is rwkv's version of low rank adaptation

        # the result has two components: a base value vector, and an offset
        # the offset is calculated by taking token shifted x and squeezing it through shrinking and expanding linear layers
        # using tanh as an activation in the middle of that sandwich
        # this offers greatly reduced cost in terms of both computation and parameters than a single dim->dim linear layer
        return self.base + self.Wb( nn.functional.tanh( self.Wa(x) ) )


# data-dependent linear interpolation
class DDLerp(nn.Module):
    def __init__(self, dim:int, dim_hidden:int):
        super().__init__()
        self.Win = nn.Linear(dim, dim, bias=False)
        self.lora = LoRA(dim, dim_hidden)

    def forward(self, x : Tensor, x_shifted_one_to_the_past : Tensor): # x (B,T,C)
        # a data-dependent linear interpolation between the current and previous token embeddings in the sequence
        # note that it is a per-channel interpolation amount, not just a single value per head

        # project the input
        y = self.Win(x)

        # linearly interpolate based on that result
        y = torch.lerp(x, x_shifted_one_to_the_past, y)

        # lora the interpolated value
        y = self.lora(y)

        # linearly interpolate again, this time based on the results of the lora
        y = torch.lerp(x, x_shifted_one_to_the_past, y)

        return y
        */

// class TimeShift_V6
// {
//     public:
//         uint32_t shiftamount = 1;

//         Tensor state;
        
//         size_t max_batch;
//         size_t max_seq;
//         size_t dims;
//         Tensor buffer;
        
//         TimeShift_V6(){
//         }

//         TimeShift_V6(safetensors& model, std::string prefix, size_t batch_size = 1){
//             // std::vector<size_t> state_size = {batch_size, 1, time_mix_r.shape[1]};
//             // std::cout << "TimeShift:" << state_size[0] << std::endl;
//             // this->state = Tensor(state_size);

            
//             // this->dims = time_mix_r.shape[0];



            
//         }

//         Tensor operator()(Tensor input){

//             if (buffer.data == nullptr || buffer.shape[1] * buffer.shape[2] < input.shape[0] * input.shape[1] || buffer.dtype != input.dtype || buffer.device != input.device){
//                 buffer = Tensor({input.shape[0],input.shape[1], input.shape[2]}, input.dtype, input.device);
//             }

//             auto batches = input.shape[0];
//             auto seq = input.shape[1];
//             Tensor output = buffer.cloneWithFalseReshape({5,batches, seq, this->dims});

//             // this->time_mix.shift(input, output, this->state, this->dims);
            
//             return output;
//         }


//         void cuda(){
//             this->state = this->state.cuda();
//             this->buffer = this->buffer.cuda();
//             // this->time_mix = this->time_mix.cuda();
//         }

// };

#endif