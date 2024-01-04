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
        RWKV_5_ATT att;
        FFN ffn;
        size_t layerid = 0;
        
        Block(safetensors& model, size_t layerID){
            layerid = layerID;
            ln1 = LayerNorm(model["blocks." + std::to_string(layerID) + ".ln1.weight"], model["blocks." + std::to_string(layerID) + ".ln1.bias"]);
            ln2 = LayerNorm(model["blocks." + std::to_string(layerID) + ".ln2.weight"], model["blocks." + std::to_string(layerID) + ".ln2.bias"]);
            att = RWKV_5_ATT(layerID, model);
            ffn = FFN(layerID, model);
        }
        Tensor operator()(Tensor x){

            auto attout = att(ln1(x), x);

            // std::cout << "attout:" << attout << std::endl;
            // attout:Tensor(0.697127, -0.114128, 0.226859, 0.0695565, , ..., 0.12287, 0.215369, 0.0816482, -0.177626, shape=(1, 18, 2048))

            auto ffnout = ffn(ln2(attout), attout);


            return ffnout;
            
        }

};