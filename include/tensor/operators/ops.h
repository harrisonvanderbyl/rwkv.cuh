

#include "tensor/operators/chevron/chevron.h"
#include "tensor/operators/reshape/reshape.h"
#include "tensor/operators/gather/gather.h"
#include "tensor/operators/normalize/normalize.h"
#include "tensor/operators/lerp/lerp.h"
#include "tensor/operators/matmul/matmul8.h"

#if defined(__CUDACC__)
#include "tensor/operators/normalize/normalize.cuh"
#include "tensor/operators/lerp/lerp.cuh"
#include "tensor/operators/matmul/matmul8.cuh"
#else
#include "tensor/operators/normalize/cpu.h"
#include "tensor/operators/lerp/cpu.h"
#include "tensor/operators/matmul/cpu.h"
#endif


#include "tensor/operators/threading/threading.h"