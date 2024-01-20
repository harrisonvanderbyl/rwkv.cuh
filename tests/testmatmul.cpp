#include <iostream>
#include "tensor/safetensors.h"

// target m512
#include <immintrin.h>

void qlerp(float* a, float*ar, float*ao, float* b, float* br, float* bo, float* mixo, __m128i* output, float* outputo){
    // outputr is equal to br
    auto o = _mm256_load_ps(ao);
    auto s = _mm256_load_ps(ar);
    auto mix = _mm256_load_ps(mixo);

    auto o2 = _mm256_load_ps(bo);
    auto s2 = _mm256_load_ps(br);
    auto y =  _mm256_load_ps(a);
    auto y2 =  _mm256_load_ps(b);
    
    // __m256 expect = ((y+o)*s)*mix + ((y2+o2)*s2)*(1.0-mix);
    auto offset = (o*s*mix + o2*s2*(1.0-mix))/s2;

    _mm_storeu_si64(output,_mm256_cvtepi32_epi8(_mm256_cvtps_epi32(y*s*mix/s2 - y2*mix + y2)));
    _mm256_store_ps(outputo, offset);
    // auto expo = (mm + offset ) * s2;
    
}

__attribute__ ((target ("avx512bf16", "avx512vl","avx2")))
int main(){
    Tensor a{{64}};
    Tensor b{{64}};
    for(size_t i = 0; i < 64; i++){
        a.get<float>(i) = -284 + 358*float(rand())/float(RAND_MAX);
        b.get<float>(i) = 128;
    }
    Tensor c{{64}};
    Tensor d{{64}};

    auto r = a.normalize(b,c,d);

   

    int8_t a[64] = {0};
    uint8_t b[64] = {0};
    int32_t out[16] = {0};
    for (size_t i = 0; i < 64; i+=1){
        a[i] = -1;
        b[i] = 1;
        if(i<16){
            out[i] = 0;
        }
    }
    auto aa = _mm512_loadu_epi8(a);
    auto bb = _mm512_loadu_epi8(b);
    auto cc = _mm512_loadu_epi32(out);
    cc =  _mm512_dpbusd_epi32(cc,bb,aa);

    std::cout << _mm512_reduce_add_epi32(cc);

    return 0;
}