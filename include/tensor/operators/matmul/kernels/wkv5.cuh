#ifndef WKV_CUH
#define WKV_CUH
#include "tensor/operators/matmul/kernels/globals.cuh"

__global__ void wkvatt(size_t TT, size_t CH, float *kk, float *vv, float *rr, float *ww, float *uu, float *ss, float *out, size_t H)
{

    // bb is batch
    // hh is head
    size_t hh = blockIdx.x;
    size_t bb = blockIdx.y;
    size_t j = threadIdx.y;
    // 1d
    uint32_t bsize = H * TT * CH;

    // 1d tensor
    uint32_t tsize = H * CH;
    // 2d tensor
    uint32_t ttsize = H * CH * CH;

    // 1d
    uint32_t hsize = CH;
    // 2d
    uint32_t hhsize = CH * CH;
    auto btimeoffset = bb * bsize;

    for (uint32_t t = 0; t < TT; t++)
    {
        auto timeoffset = btimeoffset + t * tsize;
        auto bbhsize = bb * ttsize;
        auto hoffset = hh * hsize;
        auto bhofseti = timeoffset + hoffset;
        auto bbhhsize = bbhsize + hh * hhsize;

        for (uint32_t i = 0; i < CH; i++)
        {

            uint32_t iind = bhofseti + i;
            auto hoffseti = hoffset + i;
            auto bbhhofseti = bbhhsize + i * hsize;

            float kkk = float(kk[iind]);
            float uuu = float(uu[hoffseti]);
            float rrr = float(rr[iind]);
            float www = ((double(ww[iind])));

            uint32_t jind = bhofseti + j;
            uint32_t sind = bbhhofseti + j;

            // atu = k[t,bb,hh,i]*v[t,bb,hh,j]
            float vvv = float(vv[jind]);
            float sss = ss[sind];

            // multiply kkk and vvv
            auto atu = (vvv * kkk);

            // out[t,bb,hh,j] += r[t,bb,hh,i]*(s[bb,hh,i,j] + atu*u[hh,i] )
            auto sssatuuuu = (atu * uuu + sss);
            if (i == 0)
            {
                out[jind] = 0.0f;
            }
            out[jind] += sssatuuuu * rrr;

            ss[sind] = sss * www + atu;
        }
    }
}

void wkv5_cuda_kernel(void *kk, void *vv, void *ww, void *uu, void *rr, void *ss, void *out, size_t T, size_t B, size_t C, size_t H, TENSORTYPE dtype)
{
    dim3 dimBlock(1, C / H, 1);
    dim3 dimGrid(H, B, 1);
    if (dtype == TENSORTYPE::kFLOAT_32)
        wkvatt<<<dimGrid, dimBlock>>>(T, C / H, (float *)kk, (float *)vv, (float *)rr, (float *)ww, (float *)uu, (float *)ss, (float *)out, H);
    else
        throw std::runtime_error("wkv5 not implemented for this dtype");
}

#endif