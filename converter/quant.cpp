#include <torch/extension.h>
#include "ATen/ATen.h"
#include <algorithm>


void Quantize(torch::Tensor &At, torch::Tensor &Art, torch::Tensor &Aot, torch::Tensor &Aqt, int64_t M, int64_t N, bool addResiduals = true)
{
    float *A = At.data_ptr<float>();
    double *Ar = Art.data_ptr<double>();
    double *Ao = Aot.data_ptr<double>();
    uint8_t *Aq = Aqt.data_ptr<uint8_t>();

    int64_t i, j;
    for (i = 0; i < M; i++)
    {
        double amax = (-1e9);
        double amin = (1e9);
        for (j = 0; j < N; j += 1)
        {
            double a = *(A + i * N + j);
            amax = std::max(amax, a);
            amin = std::min(amin, a);
        }
        double max = (amax);
        double min = (amin);
        double range = (max - min);
        uint BITS = 255;
        double scale = (range/BITS);
        double rangehelp = (1e44*7.13624);
        *(Ar + i)= scale;// * rangehelp;
        *(Ao + i)= min;

        // float diff = 0.0;
        for (j = 0; j < N; j += 1)
        {
            double a = *(A + i * N + j);

            double d = ((a - (min))/(scale));
            
                // std::cout << d[k] << ":" << int64_t(d[k]) << ":" << int((uint8_t)(int(d[k]))) << ":" << int((uint8_t)((unsigned int)(d[k]))) << std::endl;
            Aq[i * N + j] = (u_int8_t)((u_int32_t)(d));
            
        }

        // if (addResiduals){
        //     diff = diff / N;
        //     auto offset = (diff * scale);
        //     *(Ao + i) = *(Ao + i) + offset;
            
        // }
    }
}

// pytorch bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantize_cpu", &Quantize, "QuantizeCpu");
}

TORCH_LIBRARY(wkv5, m)
{
    m.def("quantize_cpu", Quantize);
}