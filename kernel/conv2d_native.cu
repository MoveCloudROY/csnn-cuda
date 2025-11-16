#include "common.cuh"
#include "conv2d_native.cuh"
#include "kernel.cuh"
#include <iostream>


__constant__ float d_Conv1_ker[CONV1_Co * CONV_KERNEL_K * CONV_KERNEL_K];
__constant__ float d_Conv1_b[CONV1_Co];

void conv2d_c1_k5_native_wrapper(
    float* __restrict__ x, // [N, 1, 28, 28]
    float*  w, // [Co=8, 1, K=5, K=5]
    float*  b, // [Co=8]
    float* __restrict__ y,       // [N, Co=8, 24, 24]
    int N, int Co
) {
    cudaMemcpyToSymbol(d_Conv1_ker, w, CONV1_Co * CONV_KERNEL_K * CONV_KERNEL_K * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("== conv1 w initialize failed: %s\n", cudaGetErrorString(err));
    cudaMemcpyToSymbol(d_Conv1_b, b, CONV1_Co * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("== conv1 b initialize failed: %s\n", cudaGetErrorString(err));

    const dim3 block(CONV1_THREAD_PER_BLOCK);
    const dim3 grid(CONV1_BLOCK_M * CONV1_BLOCK_N,8, 25);
    conv2d_c1_k5_native<<<grid, block>>>(
        x,
        w,
        b,
        y,
        N,
        Co
    );
}

__inline__ __device__
float warpReduceSum(float val) {
    const unsigned FULL_MASK = 0xffffffffu;
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// __constant__ float d_Conv1_ker[CONV_KERNEL_K * CONV_KERNEL_K];
// __constant__ float d_Conv1_b[CONV1_Co];

__global__ void conv2d_c1_k5_native(
    float* __restrict__ x, // [N, 1, 28, 28]
    float* __restrict__ w, // [Co=8, 1, K=5, K=5]
    float* __restrict__ b, // [Co=8]
    float* __restrict__ y,       // [N, Co=8, 24, 24]
    int N, int Co
) {
    __shared__ __half tile[CONV1_SHARED_M * CONV1_SHARED_N];
    __shared__ __half result[CONV1_CALC_M_PER_BLOCK * CONV1_CALC_N_PER_BLOCK];
    // int by = blockIdx.y;
    for (int n = blockIdx.z; n < N; n += gridDim.z) {
        int Co_id = blockIdx.y; // Co
        // for (int n = blockIdx.x; n < N;)
        const int tx    = threadIdx.x;
        int warp_id = tx >> 5;
        int lane_id = tx & 31;
        // blockIdx
        // this block area id is (block_row, block_col)
        int block_row = blockIdx.x / CONV1_BLOCK_N;
        int block_col = blockIdx.x % CONV1_BLOCK_N;
        // this block process area(tile) begin at (g_row, g_col)
        int g_row = block_row * CONV1_CALC_M_PER_BLOCK;
        int g_col = block_col * CONV1_CALC_N_PER_BLOCK;
        // Load input tile into shared memory, ld128
        constexpr int thread_per_row = CONV1_SHARED_N / 4;
        // each thread load index in tile is (row, col)
        // each thread load 4 floats    
        int row =  tx / thread_per_row;
        int col = (tx % thread_per_row) * 4;
        // 8 * 12 / 4 = 24 threads to load total tile, so just check row
        if (row < CONV1_SHARED_M) {
            // FETCH_FLOAT4(tile[row * CONV1_SHARED_N + col]) = 
            int id = row * CONV1_SHARED_N + col;
            float4 x4 = FETCH_FLOAT4(x[n * CONV1_IN_SIZE + (g_row + row) * CONV1_Wi + (g_col + col)]);
            half2 h2_0 = __float22half2_rn(make_float2(x4.x, x4.y));
            half2 h2_1 = __float22half2_rn(make_float2(x4.z, x4.w));
            *((half2*)&tile[id]) = h2_0;
            *((half2*)&tile[id + 2]) = h2_1;
        }
        __syncthreads();
        // Compute convolution
        // for (int orow = 0; orow < CONV1_CALC_M_PER_BLOCK; ++orow) {
        //     for (int ocol = 0; ocol < CONV1_CALC_N_PER_BLOCK; ++ocol) {
                // result[orow * CONV1_CALC_N_PER_BLOCK + ocol] = __float2half(0.0f);
                // warp reduce
        // each warp compute once conv kernel mutiply
        // crow ccol [0-4]
        int warp_num = CONV1_THREAD_PER_BLOCK / warpSize;
        for (int o = warp_id; o < CONV1_CALC_M_PER_BLOCK * CONV1_CALC_N_PER_BLOCK; o += warp_num) {
            int crow = lane_id / CONV1_KENREL_SIZE;
            int ccol = lane_id % CONV1_KENREL_SIZE;
            int warp_row = o / CONV1_CALC_N_PER_BLOCK;
            int warp_col = o % CONV1_CALC_N_PER_BLOCK;
            float acc = 0.0f;
            if (crow < CONV1_KENREL_SIZE) {
                float wv = d_Conv1_ker[Co_id * CONV1_KENREL_SIZE * CONV1_KENREL_SIZE + crow * CONV1_KENREL_SIZE + ccol];
                float xv = __half2float(tile[(warp_row + crow) * CONV1_SHARED_N + warp_col + ccol]);
                acc = xv * wv;
            }
            // warp shuffle to reduce sum
            // (block_row, block_col) + ([0-4], [0-4]) -> (0,0)
            acc = warpReduceSum(acc);
            if (lane_id == 0) {
                acc += d_Conv1_b[Co_id];
                result[warp_row * CONV1_CALC_N_PER_BLOCK + warp_col] = __float2half(acc);
            }

                    
            //     } 
            // }
        }
        __syncthreads();
        
        // write to y
        for (int i = tx; i < CONV1_CALC_M_PER_BLOCK * CONV1_CALC_N_PER_BLOCK; i += blockDim.x) {
            int irow = i / CONV1_CALC_N_PER_BLOCK;
            int icol = i % CONV1_CALC_N_PER_BLOCK;
            int row = g_row + irow;
            int col = g_col + icol;
            

            float v = __half2float(result[irow * CONV1_CALC_N_PER_BLOCK + icol]);
            y[(n * Co + Co_id) * CONV1_Ho * CONV1_Wo + row * CONV1_Wo + col] = v;
        }
        __syncthreads();
    }
    
}



__global__ void conv2d_nchw_native_compress(
    float* __restrict__ x, // [N, Ci=8, Hi=12, Wi=12]
    float* __restrict__ w, // [Co=16, Ci=8, K=5, K=5]
    float* __restrict__ b, // [Co=16]
    float* __restrict__ y,       // [N, Co=16, Ho=8, Wo=8]
    int N, int Ci, int Hi, int Wi, int Co
) {

}