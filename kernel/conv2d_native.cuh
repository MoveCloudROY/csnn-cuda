#pragma once
#include "common.cuh"

#define CONV_KERNEL_K  5
#define CONV1_KENREL_SIZE 5
#define CONV1_Hi 28
#define CONV1_Wi 28
#define CONV1_IN_SIZE  (CONV1_Hi * CONV1_Wi)
#define CONV1_Ho (CONV1_Hi - CONV_KERNEL_K + 1)
#define CONV1_Wo (CONV1_Wi - CONV_KERNEL_K + 1)
#define CONV1_Co 8
#define CONV1_CALC_M_PER_BLOCK 4
#define CONV1_CALC_N_PER_BLOCK 8

#define CONV1_SHARED_M (CONV_KERNEL_K + CONV1_CALC_M_PER_BLOCK - 1)
#define CONV1_SHARED_N (CONV_KERNEL_K + CONV1_CALC_N_PER_BLOCK - 1)

#define CONV1_BLOCK_M (CONV1_Ho / CONV1_CALC_M_PER_BLOCK) //2
#define CONV1_BLOCK_N (CONV1_Wo / CONV1_CALC_N_PER_BLOCK) //3

#define CONV1_THREAD_PER_BLOCK ( 256 ) //192
#define CONV1_GRIDDIMX 8


static_assert(CONV1_SHARED_N % 4 == 0, "FETCH_FLOAT4 need aligned");
static_assert(CONV1_THREAD_PER_BLOCK >= CONV1_SHARED_M * CONV1_SHARED_N / 4 , "FETCH_FLOAT4 need aligned");





/*

3 6

32 * 25



2 - 25

6 * 6

8 * 8

32

*/




__global__ void conv2d_c1_k5_native(
    float* __restrict__ x, // [N, 1, 28, 28]
    float* __restrict__ w, // [Co=8, 1, K=5, K=5]
    float* __restrict__ b, // [Co=8]
    float* __restrict__ y,       // [N, Co=8, 24, 24]
    int N, int Co
);

void conv2d_c1_k5_native_wrapper(
    float* __restrict__ x, // [N, 1, 28, 28]
    float*  w, // [Co=8, 1, K=5, K=5]
    float*  b, // [Co=8]
    float* __restrict__ y,       // [N, Co=8, 24, 24]
    int N, int Co
);


#define CONV2_KENREL_SIZE 5
#define CONV2_Hi 12
#define CONV2_Wi 12
#define CONV2_Ci 8
#define CONV2_IN_SIZE  (CONV2_Hi * CONV2_Wi)
#define CONV2_Ho (CONV2_Hi - CONV2_KENREL_SIZE + 1)
#define CONV2_Wo (CONV2_Wi - CONV2_KENREL_SIZE + 1)
#define CONV2_Co 16


__global__ void fused_conv_kernel2(const float* input, const float* weights, const float* biases,
                                                float* spikes,
                                                int N, int C, int H, int W,
                                                int K, int R, int S,
                                                int Oh, int Ow);

__global__ void conv2d_nchw_native_compress(
    float* __restrict__ x, // [N, Ci=8, Hi=12, Wi=12]
    float* __restrict__ w, // [Co=16, Ci=8, K=5, K=5]
    float* __restrict__ b, // [Co=16]
    float* __restrict__ y,       // [N, Co=16, Ho=8, Wo=8]
    int N, int Ci, int Hi, int Wi, int Co
);

__global__ void conv2d_nchw_fuse_if_native_compress(
    float* __restrict__ x, // [N, Ci=8, Hi=12, Wi=12]
    float* __restrict__ w, // [Co=16, Ci=8, K=5, K=5]
    float* __restrict__ b, // [Co=16]
    float* __restrict__ y,       // [N, Co=16, Ho=8, Wo=8]
    float* __restrict__ v,
    int N, int Ci, int Hi, int Wi, int Co
);

/*

# 1x28x28
self.conv1 = layer.Conv2d(1, 8, 5) # in_channels: 1, out_channels: 8, kernel_size: 5
# 6x24x24
self.if1 = neuron.IFNode()
self.pool1 = layer.MaxPool2d(2, 2)
# 6x12x12

self.conv2 = layer.Conv2d(8, 16, 5) # in_channels: 8, out_channels: 16, kernel_size: 5
# 16x8x8
self.if2 = neuron.IFNode()
self.pool2 = layer.MaxPool2d(2, 2)
# 16x4x4
*/