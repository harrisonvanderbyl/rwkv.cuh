#include "png/lodepng.hpp"
#include "png/resizer.hpp"
#include "rwkv.h"
/*
{'crop_size': 336, 'do_center_crop': True, 'do_normalize': True, 'do_resize': True, 'feature_extractor_type': 'CLIPFeatureExtractor', 'image_mean': [0.48145466, 0.4578275, 0.40821073], 'image_std': [0.26862954, 0.26130258, 0.27577711], 'resample': 3, 'size': 336}
*/
#include "iostream"
#include "tensor/tensor.h"
#include "tensor/modules/layernorm.h"
#include "tensor/safetensors.h"

class VITATT
{
public:
    Linear query;
    Linear key;
    Linear value;
    Linear output;

    VITATT()
    {
    }

    VITATT(int layerID, safetensors &model)
    {
        std::string prefix = "vit.vision_model.encoder.layers." + std::to_string(layerID) + ".self_attn.";

        this->key = Linear(model, prefix + "k_proj");
        this->value = Linear(model, prefix + "v_proj");
        this->query = Linear(model, prefix + "q_proj");
        this->output = Linear(model, prefix + "out_proj");
    }
    Tensor operator()(Tensor input, Tensor residual)
    {

        auto pool = get_threadpool();

        auto k = key(input);
        auto q = query(input);
        auto v = value(input);
        pool->sync();
        std::cout << "qkv\n";
        // std::cout << k;
        // std::cout << q;
        // tensor([[[ 0.2269, -0.2298, -0.1046,  ..., -0.1792, -0.0164,  0.1183],
        //  [ 0.2416, -0.2328, -0.1157,  ..., -0.0400, -0.3378,  0.1029],
        //  [ 0.2438, -0.2339, -0.1089,  ..., -0.0357, -0.3403,  0.0979],
        //  ...,
        //  [ 0.2064, -0.2128, -0.0711,  ..., -0.1088, -0.1794,  0.1037],
        //  [ 0.2036, -0.1952, -0.0502,  ..., -0.1169, -0.1598,  0.1230],
        //  [ 0.1996, -0.2241, -0.0651,  ..., -0.1254, -0.1520,  0.1220]]],
        // std::cout << v;
        auto kk = (float *)k.data;
        auto vv = (float *)v.data;
        auto qq = (float *)q.data;
        auto qkv = Tensor({1, 577, 1024});
        qkv.empty();
        auto qkvd = (float *)qkv.data;
        for (size_t h = 0; h < 16; h++)
        {
            pool->add_job([h, kk, vv, qq, qkvd]()
                          {
                              auto kh = kk + 64 * h;
                              auto vh = vv + 64 * h;
                              auto qh = qq + 64 * h;
                              auto oh = qkvd + 64 * h;
                              Tensor temp({577}, kFLOAT_64);
                              temp.empty();
                              for (size_t i = 0; i < 577; i += 1)
                              {
                                  double expsum = 0.0;
                                  for (size_t j = 0; j < 577; j += 1)
                                  {
                                      double dp = (dot_floats(kh + j * 1024, qh + i * 1024, 64));
                                      temp[j] = dp;
                                      expsum += exp(double(dp));
                                  }
                                  for (size_t j = 0; j < 577; j += 1)
                                  {
                                        double dd = *(double *)temp[j].data;

                                        dd = exp(double(dd) - log(expsum));
                                          for (size_t m = 0; m < 64; m += 1)
                                        {
                                            *(oh + j * 1024 + m) += dd * *((vh + i * 1024) + m);
                                        }
                                  }

                                  if (h == 0 && i == 0)
                                    {
                                        std::cout << temp << "\n";
                                        /*
                                        torch.Size([577, 577]) tensor([[ 5.3109, -0.8409, -0.8697,  ..., -0.8482, -0.8860, -0.9995],
                [ 6.8292,  3.6325,  3.5638,  ...,  1.1729, -0.0546,  0.3532],
                [ 6.8299,  3.6115,  3.5477,  ...,  1.1751, -0.0435,  0.3483],
                ...,
                [ 5.8861,  1.4906,  1.4427,  ...,  0.7063, -0.0830, -0.0070],
                [ 5.5077,  0.9889,  0.9714,  ...,  0.4002,  0.7696,  0.3320],
                [ 5.5250,  1.0490,  1.0035,  ...,  0.3035,  0.1337,  0.4445]],
                                        */
                                    }
                              }
                              

                              //   for (size_t i = 0; i < 577; i += 1)
                              //   {
                              //       for (size_t j = 0; j < 577; j += 1)
                              //       {

                              //           for (size_t m = 0; m < 64; m += 1)
                              //           {
                              //               *(oh + j * 1024 + m) += *(float *)temp[i][j].data * *((vh + i * 1024) + m);
                              //           }
                              //       }
                              //   }
                          },
                          h);
        }

        // auto qkv = k.matmul(q).matmul(v);
        /*
        torch.Size([1, 577, 1024]) tensor([[[ 0.0307, -0.0128, -0.0258,  ...,  0.0082, -0.0246, -0.0115],
         [ 0.0794, -0.0674, -0.0414,  ...,  0.0541, -0.0428, -0.0266],
         [ 0.0793, -0.0672, -0.0413,  ...,  0.0542, -0.0429, -0.0264],
         ...,
         [ 0.0613, -0.0465, -0.0381,  ...,  0.0383, -0.0372, -0.0226],
         [ 0.0584, -0.0429, -0.0419,  ...,  0.0330, -0.0350, -0.0214],
         [ 0.0571, -0.0421, -0.0396,  ...,  0.0311, -0.0344, -0.0206]]],
        */
        pool->sync();
        // std::cout << qkv;
        // exit(0);
        auto xx =output(qkv);
        xx += residual;
        return xx;
    }
};
class VITFFN
{
public:
    Linear fc1;
    Linear fc2;

    VITFFN()
    {
    }

    VITFFN(int layerID, safetensors &model)
    {
        std::string prefix = "vit.vision_model.encoder.layers." + std::to_string(layerID) + ".mlp.";

        this->fc1 = Linear(model, prefix + "fc1", GELU);
        this->fc2 = Linear(model, prefix + "fc2");
    }
    Tensor operator()(Tensor input, Tensor residual)
    {

        auto pool = get_threadpool();

        auto k = fc1(input);
        pool->sync();
        auto v = fc2(k);
        v += residual;
        pool->sync();
        return v;
    }
};
class VitBlock
{
public:
    LayerNorm ln1;
    LayerNorm ln2;
    VITATT att;
    VITFFN ffn;
    size_t layerid = 0;
    TimeShift attshift;
    TimeShift ffnshift;

    VitBlock(safetensors &model, size_t layerID)
    {
        layerid = layerID;
        ln1 = LayerNorm(model["vit.vision_model.encoder.layers." + std::to_string(layerID) + ".layer_norm1.weight"], model["vit.vision_model.encoder.layers." + std::to_string(layerID) + ".layer_norm1.bias"]);
        ln2 = LayerNorm(model["vit.vision_model.encoder.layers." + std::to_string(layerID) + ".layer_norm2.weight"], model["vit.vision_model.encoder.layers." + std::to_string(layerID) + ".layer_norm2.bias"]);
        att = VITATT(layerID, model);
        ffn = VITFFN(layerID, model);
    }
    Tensor operator()(Tensor x)
    {
        // get cuda error
        check_for_errors();
        auto threadpool = get_threadpool();
        threadpool->debug(x, "start x");
        auto lx = att(ln1(x), x);

        threadpool->debug(x, "att out");
        return ffn(ln2(lx), lx);
    }
};
class ImageProcessor
{
public:
    Embedding emb;
    Linear patch_embedding;
    LayerNorm pre_layernorm;
    Linear output;
    std::vector<VitBlock> layers;

    ImageProcessor(safetensors &model)
    {
        auto keys = model.keys();
        // for(auto key : keys){
        //     std::cout << key << "\n";
        // }
        emb = Embedding(model["vit.vision_model.embeddings.position_embedding.weight"]);
        patch_embedding = Linear(model, "vit.vision_model.embeddings.patch_embedding", VITEMB);
        pre_layernorm = LayerNorm(model["vit.vision_model.pre_layrnorm.weight"], model["vit.vision_model.pre_layrnorm.bias"]);
        output = Linear(model,"proj");

        std::cout << emb.weight << "\n";
        std::cout << patch_embedding.weight << ": pemb\n";
        for (size_t i = 0; i < 24; i++)
        {
            layers.push_back(VitBlock(model, i));
        }
    };
    Tensor process_image(std::string pngfile)
    {
        std::vector<unsigned char> buffer;
        lodepng::load_file(buffer, pngfile);
        // std::cout << buffer.size() << "\n";
        std::vector<unsigned char> ubuffer;
        unsigned int w, h;
        lodepng::State s;
        lodepng::decode(ubuffer, w, h, buffer.data(), buffer.size(), LodePNGColorType::LCT_RGB, 8U);
        // std::cout << ubuffer.size() << "\n"
        //           << w << ":" << h << "\n";
        avir::CImageResizer ImageResizer(8);
        // std::vector<unsigned char> Outbuffer(336*336*3);
        Tensor Outbuffer({336, 336, 3}, kUINT_8);
        ImageResizer.resizeImage(ubuffer.data(), int(w), int(h), 0, (unsigned char *)Outbuffer.data, 336, 336, 3, 0);

        std::vector<unsigned char> fbuf;
        lodepng::encode(fbuf, std::vector<unsigned char>((unsigned char *)Outbuffer.data, ((unsigned char *)Outbuffer.data) + Outbuffer.get_element_count()), 336, 336, LodePNGColorType::LCT_RGB, 8U);
        lodepng::save_file(fbuf, "testout.png");
        Tensor Processed({576, 3, 14, 14});
        float image_mean[3] = {0.48145466, 0.4578275, 0.40821073};
        float image_std[3] = {0.26862954, 0.26130258, 0.27577711};
        size_t pos = 0;
        //  tensor([ 0.0155,  0.2856, -0.1434,  ...,  0.0176, -0.3926, -0.2852]

        for (size_t j = 0; j < 336; j += 14)
        {
            for (size_t i = 0; i < 336; i += 14)
            {
                for (size_t k = 0; k < 3; k++)
                {
                    auto toprocess = Processed[pos];
                    for (size_t ii = 0; ii < 14; ii++)
                    {
                        for (size_t jj = 0; jj < 14; jj++)
                        {
                            toprocess[k][ii][jj] = (float(*(uint8_t *)Outbuffer[i + ii][j + jj * 336][k].data) / 255 - image_mean[k]) / image_std[k];
                        }
                    }
                }
                pos += 1;
            }
        }
        std::cout << Processed;
        auto cff = Processed.cloneWithFalseReshape({1, 576, 588});
        std::cout << "cff:" << cff;
        auto embb = emb();
        std::cout << embb;
        auto pemp = patch_embedding(cff, embb);
        auto pemp2 = pre_layernorm(pemp);

        for (auto layer : layers)
        {
            pemp2 = layer(pemp2);
        }

        return output(pemp2);

        // lodepng::save_file(buffer,"./testout.png");
    };
};