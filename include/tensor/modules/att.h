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
        Tensor time_mix_x;
        Tensor time_mix_w;
        Tensor time_mix_w1;
        Tensor time_mix_w2;
        Tensor time_decay_w1;
        Tensor time_decay_w2;
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
        bool v6 = false;

        RWKV_5_ATT(){
        }
        
        RWKV_5_ATT(int layerID, safetensors& model){
            // std::cout << "RWKV_5_ATTcreate:" << layerID << std::endl;
            std::string prefix = "blocks." + std::to_string(layerID) + ".att.";
            this->layer = layerID;

            auto point = "mix";
            if (!model.contains(prefix + "time_"+point+"_k")){
                point = "maa";
                v6 = true;
                // time_mix_w1 = Linear(model, prefix + "time_"+point+"_w1");
                // time_mix_w2 = Linear(model, prefix + "time_"+point+"_w2");
                time_mix_x = model[prefix + "time_"+point+"_x"][0][0];
                time_mix_w1 = model[prefix + "time_"+point+"_w1"];
                time_mix_w2 = model[prefix + "time_"+point+"_w2"];
                time_decay_w1 = model[prefix + "time_decay_w1"];
                time_decay_w2 = model[prefix + "time_decay_w2"];
                time_mix_w = model[prefix + "time_"+point+"_w"][0][0];
                std::cout << "v6" << std::endl;
                std::cout << time_mix_w1 << std::endl;
                std::cout << time_mix_w2 << std::endl;
                std::cout << time_decay_w1 << std::endl;
                std::cout << time_decay_w2 << std::endl;

            }

            this->time_mix_k = model[prefix + "time_"+point+"_k"][0][0];
            this->time_mix_v = model[prefix + "time_"+point+"_v"][0][0];
            this->time_mix_r = model[prefix + "time_"+point+"_r"][0][0];
            this->time_mix_g = model[prefix + "time_"+point+"_g"][0][0];



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

            Tensor t_mix_k;
            Tensor t_mix_v;
            Tensor t_mix_r;
            Tensor t_mix_g;

            Tensor decay;
            
            if (v6){
                auto xxx = time_mix_x.lerp(xx, input, cbuf, v6);
                Tensor bufmak = Tensor({xx.shape[0],xx.shape[1],time_mix_w1.shape[0]}, xx.dtype, xx.device);

                bufmak.empty();
                Tensor bufmak2 = Tensor({5, xx.shape[0], xx.shape[1], xx.shape[2]}, xx.dtype, xx.device);
                bufmak2.empty();
                xxx = time_mix_w1.matmul(xxx, bufmak);
                
                xxx = xxx.tahn();
                
                for (int i = 0; i < xx.shape[0]; i++){
                    for (int j = 0; j < xx.shape[1]; j++){

                        bufmak2[0][i][j].copyfrom(time_mix_w);
                        bufmak2[1][i][j].copyfrom(time_mix_k);
                        bufmak2[2][i][j].copyfrom(time_mix_v);
                        bufmak2[3][i][j].copyfrom(time_mix_r);
                        bufmak2[4][i][j].copyfrom(time_mix_g);
                    }
                }

                auto clnd = Tensor({
                    5, 
                    xxx.shape[0], 
                    xxx.shape[1], 
                    xxx.shape[2]/5
                
                }, xxx.dtype, xxx.device);
                clnd.empty();
                auto rsh = xxx.reshape({xxx.shape[0],xxx.shape[1], 5, xxx.shape[2]/5});
                
                for (int i = 0; i < xxx.shape[0]; i++){
                    for (int j = 0; j < xxx.shape[1]; j++){
                        clnd[0][i][j].copyfrom(rsh[i][j][0].reshape({this->n_head}));
                        clnd[1][i][j].copyfrom(rsh[i][j][1].reshape({this->n_head}));
                        clnd[2][i][j].copyfrom(rsh[i][j][2].reshape({this->n_head}));
                        clnd[3][i][j].copyfrom(rsh[i][j][3].reshape({this->n_head}));
                        clnd[4][i][j].copyfrom(rsh[i][j][4].reshape({this->n_head}));
                    }
                }

                auto t_mix_w = time_mix_w2[0].matmul( clnd[0] , bufmak2[0]);
                t_mix_k = time_mix_w2[1].matmul( clnd[1] , bufmak2[1]);
                t_mix_v = time_mix_w2[2].matmul( clnd[2] , bufmak2[2]);
                t_mix_r = time_mix_w2[3].matmul( clnd[3] , bufmak2[3]);
                t_mix_g = time_mix_w2[4].matmul( clnd[4] , bufmak2[4]);

                decay = t_mix_w.lerp(xx, input, cbuf, v6);

                auto temput = Tensor({xx.shape[0],xx.shape[1], time_decay_w1.shape[0]}, xx.dtype, xx.device);
                temput.empty();
                temput = time_decay_w1.matmul(decay, temput);
                decay = temput.tahn();
                auto temput2 = Tensor({xx.shape[0],xx.shape[1], time_decay_w2.shape[0]}, xx.dtype, xx.device);
                temput2.empty();
                for (int i = 0; i < xx.shape[0]; i++){
                    for (int j = 0; j < xx.shape[1]; j++){
                        temput2[i][j].copyfrom(time_decay.reshape({time_decay.shape[0]*time_decay.shape[1]*time_decay.shape[2]}));
                    }
                }
                decay = time_decay_w2.matmul(decay, temput2);
                for (int i = 0; i < decay.get_element_count(); i++){
                    float a = flp(decay.data)[i];
                    flp(decay.data)[i] = exp(-exp(a));
                }
                // std::cout << decay << std::endl;
            }else{
                t_mix_k = time_mix_k;
                t_mix_v = time_mix_v;
                t_mix_r = time_mix_r;
                t_mix_g = time_mix_g;
                decay = this->time_decay;
            }

            
            auto kr = t_mix_k.lerp(xx, input, cbuf,v6);
            auto k = this->key(kr);      
            auto vr = t_mix_v.lerp(xx, input, cbuf,v6);
            auto v = this->value(vr);
            auto rr = t_mix_r.lerp(xx, input, cbuf,v6);
            auto r = this->receptance(rr);
            auto gr = t_mix_g.lerp(xx, input, cbuf,v6);
            auto gv = this->gate(gr);

    
            auto xm = this->state.wkv5(r,k,v,decay,this->time_faaaa, cbuf);

       
            auto xxa = this->ln_x(xm);


            auto gvo = gv.swishmul(xxa);

            
            return this->output(gvo, residual);
        }

};