#ifndef WKV_CUH
#define WKV_CUH
#include "tensor/operators/matmul/kernels/globals.cuh"
template <typename T>
__global__ void wkvatt(size_t TT, size_t CH, T *kk, T *vv, T *rr, T *ww, T *uu, float *ss, T *out, size_t H)
{

    // bb is batch
    // hh is head
    size_t bb = blockIdx.x ;
    size_t hh = threadIdx.x;
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

    for (uint32_t t = 0; t < TT; t++)
    {
        for (uint32_t i = 0; i < CH; i++)
        {
            auto btimeoffset = bb * bsize;
            auto timeoffset = btimeoffset + t * tsize;
            auto bbhsize = bb * ttsize;

            auto hoffset = hh * hsize;
            auto bhofseti = timeoffset + hoffset;
            auto bbhhsize = bbhsize + hh * hhsize;

            uint32_t iind = bhofseti + i;
            auto hoffseti = hoffset + i;
            auto bbhhofseti = bbhhsize + i * hsize;

            float kkk = float(kk[iind]);
            float uuu = float(uu[hoffseti]);
            float rrr = float(rr[iind]);
            float www = float(ww[hoffseti]);

            for (uint32_t j = 0; j < CH; j += 1)
            {
                uint32_t jind = bhofseti + j;
                uint32_t sind = bbhhofseti + j;

                // atu = k[t,bb,hh,i]*v[t,bb,hh,j]
                float vvv = float(vv[jind]);
                float sss = ss[sind];

                // multiply kkk and vvv
                auto atu = (vvv * kkk);

                // out[t,bb,hh,j] += r[t,bb,hh,i]*(s[bb,hh,i,j] + atu*u[hh,i] )
                auto sssatuuuu = (atu * uuu + sss);

                out[jind] += T(sssatuuuu * rrr);

                ss[sind] = sss * www + atu;
            }
        }
    }
}

#endif 