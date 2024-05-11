#include "tensor/tensor.h"
#include "tensor/modules/layernorm.h"
#include "tensor/safetensors.h"
#include "tensor/modules/att.h"
#include "tensor/modules/ffn.h"
class Block
{
    public:
        LayerNorm ln1;
        LayerNorm ln2;
        Attention att;
        FeedForward ffn;
        size_t layerid = 0;
        TimeShift attshift;
        TimeShift ffnshift;
        
        Block(safetensors& model, size_t layerID){
            layerid = layerID;
            ln1 = LayerNorm(model["blocks." + std::to_string(layerID) + ".ln1.weight"], model["blocks." + std::to_string(layerID) + ".ln1.bias"]);
            ln2 = LayerNorm(model["blocks." + std::to_string(layerID) + ".ln2.weight"], model["blocks." + std::to_string(layerID) + ".ln2.bias"]);
            attshift = TimeShift(model["blocks." + std::to_string(layerID) + ".attshift.time_mix"]);
            att = Attention(layerID, model);
            ffnshift = TimeShift(model["blocks." + std::to_string(layerID) + ".ffnshift.time_mix"]);
            ffn = FeedForward(layerID, model);
        }
        Tensor operator()(Tensor x){
            // get cuda error
            check_for_errors();
            auto threadpool = get_threadpool();
            auto attout = att(attshift(ln1(x)), x);
            auto ffnout = ffn(ffnshift(ln2(attout)), attout);
            return ffnout;
            
        }

};