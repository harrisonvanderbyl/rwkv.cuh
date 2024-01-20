

#include "tensor/operators/chevron/chevron.h"
#include "tensor/operators/reshape/reshape.h"
#include "tensor/operators/gather/gather.h"
#include "tensor/operators/normalize/normalize.h"
#include "tensor/operators/relusquare/relusquare.h"
#include "tensor/operators/sigmoidmul/sigmoidmul.h"
#include "tensor/operators/lerp/lerp.h"
#include "tensor/operators/swishmul/swishmul.h"
#include "tensor/operators/matmul/matmul8.h"
#include "tensor/operators/tahn/tahn.h"

#if defined(__CUDACC__)
#include "tensor/operators/swishmul/swishmul.cuh"
#include "tensor/operators/sigmoidmul/sigmoidmul.cuh"
#include "tensor/operators/relusquare/relusquare.cuh"
#include "tensor/operators/normalize/normalize.cuh"
#include "tensor/operators/lerp/lerp.cuh"
#include "tensor/operators/matmul/matmul8.cuh"
#endif // __CUDACC__