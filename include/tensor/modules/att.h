#include "tensor/tensor.h"

#include "tensor/safetensors.h"
#include "tensor/modules/timeshift.h"
#include "tensor/modules/linear.h"
#include "tensor/modules/layernorm.h"
class RWKV_5_ATT
{
    public:
        uint32_t head_size = 64;
        uint32_t n_head; 
        TimeShift timeshift;
        Tensor time_mix_k;
        Tensor time_mix_v;
        Tensor time_mix_r;
        Tensor time_mix_g;
        Tensor time_decay;
        Tensor time_faaaa;
        Tensor state;
        Linear receptance;
        Linear key;
        Linear value;
        Linear gate;
        Linear output;
        LayerNorm ln_x;
        Tensor buffer;
        int layer = 0;

        RWKV_5_ATT(){
        }
        
        RWKV_5_ATT(int layerID, safetensors& model){
            // std::cout << "RWKV_5_ATTcreate:" << layerID << std::endl;
            std::string prefix = "blocks." + std::to_string(layerID) + ".att.";
            this->layer = layerID;
            this->time_mix_k = model[prefix + "time_mix_k"][0][0];
            this->time_mix_v = model[prefix + "time_mix_v"][0][0];
            this->time_mix_r = model[prefix + "time_mix_r"][0][0];
            this->time_mix_g = model[prefix + "time_mix_g"][0][0];

            auto dims = this->time_mix_k.shape[0];

            
            this->n_head = dims/this->head_size;
            this->state = Tensor({1, this->n_head , this->head_size, this->head_size});
            
            this->time_decay = model[prefix + "time_decay"];
            this->time_faaaa = model[prefix + "time_faaaa"];
            
            this->timeshift = TimeShift(dims);

            this->receptance = Linear(model, prefix + "receptance");
            this->key = Linear(model, prefix + "key");
            this->value = Linear(model, prefix + "value");
            this->gate = Linear(model, prefix + "gate");
            this->output = Linear(model, prefix + "output");
            this->ln_x = LayerNorm(model[prefix + "ln_x.weight"], model[prefix + "ln_x.bias"], n_head, 64e-5);
            
        }



        Tensor operator()(Tensor input, Tensor residual){


            if(buffer.data == nullptr || buffer.shape[0] * buffer.shape[1] < input.shape[0] * input.shape[1] || buffer.dtype != input.dtype || buffer.device != input.device){
                buffer = Tensor({input.shape[0],input.shape[1], input.shape[2]}, input.dtype, input.device);
            }

            auto cbuf = buffer.cloneWithFalseReshape({input.shape[0],input.shape[1], input.shape[2]});
            
            auto xx = this->timeshift(input);
            
            
            
            auto kr = this->time_mix_k.lerp(xx, input, cbuf);
            auto k = this->key(kr);
            auto vr = this->time_mix_v.lerp(xx, input, cbuf);
            auto v = this->value(vr);
            auto rr = this->time_mix_r.lerp(xx, input, cbuf);
            auto r = this->receptance(rr);
            auto gr = this->time_mix_g.lerp(xx, input, cbuf);
            auto gv = this->gate(gr);

    
            auto xm = this->state.wkv5(r,k,v,this->time_decay,this->time_faaaa, cbuf);

           
       
            auto xxa = this->ln_x(xm);


            auto gvo = gv.swishmul(xxa);

            
            return this->output(gvo, residual);
        }

};