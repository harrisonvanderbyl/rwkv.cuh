#include "png/lodepng.hpp"
#include "png/resizer.hpp"
#include "rwkv.h"
/*
{'crop_size': 336, 'do_center_crop': True, 'do_normalize': True, 'do_resize': True, 'feature_extractor_type': 'CLIPFeatureExtractor', 'image_mean': [0.48145466, 0.4578275, 0.40821073], 'image_std': [0.26862954, 0.26130258, 0.27577711], 'resample': 3, 'size': 336}
*/
#include "iostream"
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
    Tensor Processed({3, 336, 336});
    float image_mean[3] = {0.48145466, 0.4578275, 0.40821073};
    float image_std[3] = {0.26862954, 0.26130258, 0.27577711};
    for (size_t i = 0; i < 336; i++)
    {
        for (size_t j = 0; j < 336; j++)
        {
            for (size_t k = 0; k < 3; k++)
            {
                Processed[k][i][j] = (float(*(uint8_t*)Outbuffer[i][j][k].data)/255 - image_mean[k] )/image_std[k];
            }
        }
    }
    // std::cout << Processed;
    return Processed;

    // lodepng::save_file(buffer,"./testout.png");
}