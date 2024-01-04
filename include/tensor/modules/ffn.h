#include "tensor/tensor.h"

#include "tensor/safetensors.h"
#include "tensor/modules/timeshift.h"
#include "tensor/modules/linear.h"
class FFN
{
    public:
        uint32_t head_size = 64;
        uint32_t n_head; 
        TimeShift timeshift;
        Tensor time_mix_k;
        Tensor time_mix_r;
        Linear receptance;
        Linear key;
        Linear value;
        Tensor buffer;

        FFN(){
        }
        
        FFN(int layerID, safetensors& model){
            std::string prefix = "blocks." + std::to_string(layerID) + ".ffn.";

            this->time_mix_k = model[prefix + "time_mix_k"][0][0];
            this->time_mix_r = model[prefix + "time_mix_r"][0][0];

            auto dims = this->time_mix_k.shape[0];
            // std::cout << "dims:" << dims << std::endl;

            this->timeshift = TimeShift(dims);

            this->receptance = Linear(model, prefix + "receptance");
            this->key = Linear(model, prefix + "key");
            this->value = Linear(model, prefix + "value");
        }
        Tensor operator()(Tensor input, Tensor residual){

            if (buffer.data == nullptr || buffer.shape[0] * buffer.shape[1] < input.shape[0] * input.shape[1] || buffer.dtype != input.dtype || buffer.device != input.device){
                buffer = Tensor({input.shape[0],input.shape[1], input.shape[2]}, input.dtype, input.device);
            }

            auto cbuf = buffer.cloneWithFalseReshape({input.shape[0],input.shape[1], input.shape[2]});
            
            auto xx = timeshift(input);

            auto kr = time_mix_k.lerp(xx, input, cbuf);
            auto k = key(kr);

           
            auto rr = this->time_mix_r.lerp(xx, input, cbuf);
            auto r = this->receptance(rr);

            auto krs = k.relusquared();

            auto v = this->value(krs); 

            return r.sigmoidmul(v, residual);

        }

};