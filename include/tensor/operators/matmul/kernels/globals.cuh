#ifndef GLOBALCUDA_H
#define GLOBALCUDA_H
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "tensor/tensor.h"

#define BLOCK_SIZE 1
#define LOOPSIZE 32

// #define MM8_ONE_JTILE 8
#define MM8_ONE_JSPLIT 32
#define MM8_ONE_TILE 32
// #define BLOCKSPLIT 8


#endif // GLOBALCUDA_H