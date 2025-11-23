#include "common.cuh"
#include "conv2d_native.cuh"
#include "kernel.cuh"
#include <iostream>


__constant__ float d_Conv1_ker[CONV1_Co * CONV_KERNEL_K * CONV_KERNEL_K];
__constant__ float d_Conv1_b[CONV1_Co];


static inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

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

    // const dim3 block(CONV1_THREAD_PER_BLOCK);
    // const dim3 grid(CONV1_BLOCK_M * CONV1_BLOCK_N,8, 25);

    const dim3 block(16, 16, 1);
    const dim3 grid(div_up(24, block.x), div_up(24, block.y), N * Co);
    
    size_t smem = 
            (block.x + CONV1_KENREL_SIZE + CONV1_CALC_M_PER_BLOCK - 2) 
        * (block.y + CONV1_KENREL_SIZE + CONV1_CALC_N_PER_BLOCK - 2) 
        * sizeof(float);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    conv2d_c1_k5_native<<<grid, block, smem>>>(
        x,
        w,
        b,
        y,
        N,
        Co
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("== conv2d_c1_k5_native time: %f ms\n", milliseconds);
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
//     extern __shared__ float tile[];
//     const int               tileW = blockDim.x + CONV1_KENREL_SIZE + CONV1_CALC_M_PER_BLOCK - 2;
//     const int               tileH = blockDim.y + CONV1_KENREL_SIZE + CONV1_CALC_N_PER_BLOCK - 2;
//     const int               tx    = threadIdx.x;
//     const int               ty    = threadIdx.y;
//     const int               ow_st    = blockIdx.x * blockDim.x + tx;
//     const int               oh_st    = blockIdx.y * blockDim.y + ty;
//     const int               ow_ed    = blockIdx.x * blockDim.x + tx + CONV1_CALC_M_PER_BLOCK - 1;
//     const int               oh_ed    = blockIdx.y * blockDim.y + ty + CONV1_CALC_N_PER_BLOCK - 1;
//     const int               z     = blockIdx.z;
//     const int               oc    = z % Co;
//     const int               n     = z / Co;
//     if (n >= N)
//         return;
//     const int ow0 = blockIdx.x * blockDim.x;
//     const int oh0 = blockIdx.y * blockDim.y;
//     for (int yy = ty; yy < tileH; yy += blockDim.y) {
//         int ih = oh0 + yy;
//         for (int xx = tx; xx < tileW; xx += blockDim.x) {
//             int   iw = ow0 + xx;
//             float v  = 0.0f;
//             if (ih >= 0 && ih < CONV1_Hi && iw >= 0 && iw < CONV1_Wi) {
//                 const int x_idx = (n * CONV1_Hi + ih) * CONV1_Wi + iw;
//                 v               = __ldg(x + x_idx);
//             }
//             tile[yy * tileW + xx] = v;
//         }
//     }
//     __syncthreads();
//     if (oh < CONV1_Ho && ow < CONV1_Wo) {
//         const float* w_oc = w + oc * (CONV1_KENREL_SIZE * CONV1_KENREL_SIZE);
//         float        acc  = __ldg(b + oc);
// #pragma unroll
//         for (int ky = 0; ky < CONV1_KENREL_SIZE; ++ky) {
// #pragma unroll
//             for (int kx = 0; kx < CONV1_KENREL_SIZE; ++kx) {
//                 float xv = tile[(ty + ky) * tileW + (tx + kx)];
//                 float ww = __ldg(w_oc + ky * CONV1_KENREL_SIZE + kx);
//                 acc      = fmaf(xv, ww, acc);
//             }
//         }
//         const int y_idx = ((n * Co + oc) * CONV1_Ho + oh) * CONV1_Wo + ow;
//         y[y_idx]        = acc;
//     }
}


__global__ void fused_conv_kernel2(const float* input, const float* weights, const float* biases,
                                                float* spikes,
                                                int N, int C, int H, int W,
                                                int K, int R, int S,
                                                int Oh, int Ow) {

    // 静态常量
    const int INPUT_C = 8;
    const int INPUT_H = 12;
    const int INPUT_W = 12;
    const int FILTER_K = 16;
    const int FILTER_R = 5;
    const int FILTER_S = 5;

    // ===== 共享内存声明 =====
    // 1. 为输入数据创建双缓冲（Ping-Pong Buffer）
    //    大小为 [2][Height][Width]，用于流水线操作
    __shared__ float s_input_pingpong[2][INPUT_H][INPUT_W];
    
    // 2. 权重和偏置一次性加载，因为它们在整个卷积过程中被复用
    __shared__ float s_weights[FILTER_K][INPUT_C][FILTER_R][FILTER_S];
    __shared__ float s_biases[FILTER_K];
    
    // 线程和块索引
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y; // 64
    int b = blockIdx.z;

    // ===== 1. 初始数据加载 (非流水线部分) =====
    // 加载全部权重和偏置，这部分只执行一次
    // 加载权重 (2400 floats = 600 float4s)
    float4* s_weights_f4 = (float4*)&s_weights[0][0][0][0];
    const float4* global_weights_f4 = (const float4*)weights;
    for (int i = tid; i < 800; i += threads_per_block) {
        s_weights_f4[i] = global_weights_f4[i];
    }

    // 加载偏置
    if (tid < FILTER_K) {
        s_biases[tid] = biases[tid];
    }
    
    // ===== 2. 软件流水线 (Software Pipelining) =====
    
    // 寄存器累加器
    float accum[FILTER_K] = {0.0f};

    // --- 流水线启动 (Prologue) ---
    // 在主循环开始前，预加载第一个输入切片 (in_c = 0) 到缓冲区 0
    const int SLICE_SIZE_F4 = (INPUT_H * INPUT_W) / 4; // 12*12/4 = 36
    const float4* global_input_base_f4 = (const float4*)(input + (long long)b * C * H * W);
    
    float4* s_input_slice0_f4 = (float4*)&s_input_pingpong[0][0][0];
    for (int i = tid; i < SLICE_SIZE_F4; i += threads_per_block) {
        s_input_slice0_f4[i] = global_input_base_f4[i];
    }

    // 同步：确保权重、偏置和第一个输入切片都已加载完毕
    __syncthreads();

    // --- 流水线主循环 (Steady State) ---
    // 使用 #pragma unroll 1 明确告诉编译器不要展开此循环，以保护流水线结构
    #pragma unroll 1
    for (int in_c = 0; in_c < INPUT_C; ++in_c) {
        // 确定当前计算使用的缓冲区和下一个加载要用的缓冲区
        int current_buf_idx = in_c % 2;
        int next_buf_idx = (in_c + 1) % 2;

        // --- STAGE 1: 异步预取 (Asynchronous Prefetch) ---
        // 在计算当前切片的同时，加载下一个输入切片 (如果存在)
        if (in_c < INPUT_C - 1) {
            const float4* global_next_input_f4 = global_input_base_f4 + (long long)(in_c + 1) * SLICE_SIZE_F4 * 4 / 4; // in elements of float4
            float4* s_input_next_slice_f4 = (float4*)&s_input_pingpong[next_buf_idx][0][0];
            for (int i = tid; i < SLICE_SIZE_F4; i += threads_per_block) {
                s_input_next_slice_f4[i] = global_next_input_f4[i];
            }
        }

        // --- STAGE 2: 计算 (Compute) ---
        // 使用已加载好的当前切片进行计算
        int out_x = blockIdx.x * blockDim.x + threadIdx.x;
        int out_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (b < N && out_y < Oh && out_x < Ow) {
            #pragma unroll
            for (int ky = 0; ky < FILTER_R; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < FILTER_S; ++kx) {
                    int in_y = out_y + ky;
                    int in_x = out_x + kx;
                    
                    float input_val = s_input_pingpong[current_buf_idx][in_y][in_x];

                    #pragma unroll
                    for (int out_c = 0; out_c < FILTER_K; ++out_c) {
                        accum[out_c] += input_val * s_weights[out_c][in_c][ky][kx];
                    }
                }
            }
        }
        
        // --- STAGE 3: 同步 (Synchronize) ---
        // 这个同步点是流水线的核心：
        // 1. 保证当前(in_c)的计算已完成，才能进入下一次循环。
        // 2. 保证下一个(in_c+1)的输入切片已完全加载，以备下次循环的计算阶段使用。
        __syncthreads();
    }

    // ===== 3. 计算收尾与写回 (Epilogue) =====
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < N && out_y < Oh && out_x < Ow) {
        long long output_base_idx = (long long)b * K * Oh * Ow + (long long)out_y * Ow + out_x;

        #pragma unroll
        for (int out_c = 0; out_c < FILTER_K; ++out_c) {
            float conv_output = accum[out_c] + s_biases[out_c];
            long long element_idx = output_base_idx + (long long)out_c * Oh * Ow;
            
            // float Vmem = membrane_potential[element_idx];
            // Vmem += conv_output;
            // float spike = (Vmem >= 1.0f) ? 1.0f : 0.0f;

            spikes[element_idx] = conv_output;
            // membrane_potential[element_idx] = Vmem * (1.0f - spike);
        }
    }
}



__global__ void conv2d_nchw_native_compress(
    float* __restrict__ x, // [N, Ci=8, Hi=12, Wi=12]
    float* __restrict__ w, // [Co=16, Ci=8, K=5, K=5]
    float* __restrict__ b, // [Co=16]
    float* __restrict__ y, // [N, Co=16, Ho=8, Wo=8]
    int N, int Ci, int Hi, int Wi, int Co
) {
    __shared__ float s_tile[2][CONV2_Hi][CONV2_Wi];
    __shared__ float s_weights[CONV2_Co][CONV2_Ci][CONV2_KENREL_SIZE][CONV2_KENREL_SIZE];
    __shared__ float s_bias[CONV2_Co];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;
    const int batch = blockIdx.z;
    
    // load weights and bias 
    constexpr int weight_ld4_cnt = (CONV2_Co * CONV2_Ci * CONV2_KENREL_SIZE * CONV2_KENREL_SIZE) / 4;
    float4* s_weight_base = ((float4*)&s_weights[0][0][0][0]);
    float4* global_weight_base = ((float4*)w);
    for (int i = tid; i < weight_ld4_cnt; i += threads_per_block) {
        s_weight_base[i] = global_weight_base[i];
    }
    if (tid < CONV2_Co) {
        s_bias[tid] = b[tid];
    }

    // compute pipeline 

    // 1. prologue 
    constexpr int tile_ld4_cnt = (CONV2_Hi * CONV2_Wi) / 4; // 12*12/4 = 36
    const float4* global_input_base = (const float4*)(x + (size_t)batch * Ci * Hi * Wi);
    float4* s_tile_base = (float4*)&s_tile[0][0][0];
    for (int i = tid; i < tile_ld4_cnt; i += threads_per_block) {
        s_tile_base[i] = global_input_base[i];
    }
    __syncthreads();
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int curr_buf_idx = 1;
    int next_buf_idx = 0;
    float acc[CONV2_Co] = {0.f};
    // 2. pipeline cycle
    #pragma unroll 1
    for (int in_c = 0; in_c < CONV2_Ci; ++in_c) {
        next_buf_idx = curr_buf_idx;
        curr_buf_idx = curr_buf_idx ^ 1;
        // prefetch
        if (in_c < CONV2_Ci - 1) {
            const float4* global_input_ptr = global_input_base + (size_t)(in_c + 1) * tile_ld4_cnt;
            float4* s_next_tile_ptr = (float4*)&s_tile[next_buf_idx][0][0];
            for (int i = tid; i < tile_ld4_cnt; i += threads_per_block) {
                s_next_tile_ptr[i] = global_input_ptr[i];
            }
        }
        
        // compute
        if (out_x < CONV2_Wo && out_y < CONV2_Ho){
            #pragma unroll
            for (int ky = 0; ky < CONV2_KENREL_SIZE; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < CONV2_KENREL_SIZE; ++kx) {
                    int in_y = out_y + ky;
                    int in_x = out_x + kx;
                    float input_val = s_tile[curr_buf_idx][in_y][in_x];
                    #pragma unroll
                    for (int out_c = 0; out_c < CONV2_Co; ++out_c) {
                        float weight_val = s_weights[out_c][in_c][ky][kx];
                        acc[out_c] += input_val * weight_val;
                        // atomicAdd(&y[((size_t)batch * Co + out_c) * CONV2_Ho * CONV2_Wo + out_y * CONV2_Wo + out_x], input_val * weight_val);
                    }
                }
            }
        }
        // sync prefetch & compute
        __syncthreads();
    }
    // 3. epologue
    if (out_x < CONV2_Wo && out_y < CONV2_Ho){
        size_t y_base_offset = (size_t)batch * Co * CONV2_Ho * CONV2_Wo + out_y * CONV2_Wo + out_x;
        for (int out_c = 0; out_c < CONV2_Co; ++out_c) {
            float conv_output = acc[out_c] + s_bias[out_c];
            size_t y_channel_offset = y_base_offset + (size_t)out_c * CONV2_Ho * CONV2_Wo;
            y[y_channel_offset] = conv_output;
        }
    }
}




__global__ void conv2d_nchw_fuse_if_native_compress(
    float* __restrict__ x, // [N, Ci=8, Hi=12, Wi=12]
    float* __restrict__ w, // [Co=16, Ci=8, K=5, K=5]
    float* __restrict__ b, // [Co=16]
    float* __restrict__ y, // [N, Co=16, Ho=8, Wo=8]
    float* __restrict__ v,
    int N, int Ci, int Hi, int Wi, int Co
) {
    __shared__ float s_tile[2][CONV2_Hi][CONV2_Wi];
    __shared__ float s_weights[CONV2_Co][CONV2_Ci][CONV2_KENREL_SIZE][CONV2_KENREL_SIZE];
    __shared__ float s_bias[CONV2_Co];
    __shared__ float s_v[CONV2_Co][CONV2_Ho][CONV2_Wo];    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y;
    const int batch = blockIdx.z;
    
    // load weights and bias 
    constexpr int weight_ld4_cnt = (CONV2_Co * CONV2_Ci * CONV2_KENREL_SIZE * CONV2_KENREL_SIZE) / 4;
    float4* s_weight_base = ((float4*)&s_weights[0][0][0][0]);
    float4* global_weight_base = ((float4*)w);
    for (int i = tid; i < weight_ld4_cnt; i += threads_per_block) {
        s_weight_base[i] = global_weight_base[i];
    }
    float4* s_v_base = ((float4*)&s_v[0][0][0]);
    float4* global_v_base = ((float4*)v + (size_t)batch * Co * CONV2_Ho * CONV2_Wo / 4);
    for (int i = tid; i < (CONV2_Co * CONV2_Ho * CONV2_Wo) / 4; i += threads_per_block) {
        s_v_base[i] = global_v_base[i];
    }
    if (tid < CONV2_Co) {
        s_bias[tid] = b[tid];
    }

    // compute pipeline 

    // 1. prologue 
    constexpr int tile_ld4_cnt = (CONV2_Hi * CONV2_Wi) / 4; // 12*12/4 = 36
    const float4* global_input_base = (const float4*)(x + (size_t)batch * Ci * Hi * Wi);
    float4* s_tile_base = (float4*)&s_tile[0][0][0];
    for (int i = tid; i < tile_ld4_cnt; i += threads_per_block) {
        s_tile_base[i] = global_input_base[i];
    }
    __syncthreads();
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int curr_buf_idx = 1;
    int next_buf_idx = 0;
    float acc[CONV2_Co] = {0.f};
    // 2. pipeline cycle
    #pragma unroll 1
    for (int in_c = 0; in_c < CONV2_Ci; ++in_c) {
        next_buf_idx = curr_buf_idx;
        curr_buf_idx = curr_buf_idx ^ 1;
        // prefetch
        if (in_c < CONV2_Ci - 1) {
            const float4* global_input_ptr = global_input_base + (size_t)(in_c + 1) * tile_ld4_cnt;
            float4* s_next_tile_ptr = (float4*)&s_tile[next_buf_idx][0][0];
            for (int i = tid; i < tile_ld4_cnt; i += threads_per_block) {
                s_next_tile_ptr[i] = global_input_ptr[i];
            }
        }
        
        // compute
        if (out_x < CONV2_Wo && out_y < CONV2_Ho){
            #pragma unroll
            for (int ky = 0; ky < CONV2_KENREL_SIZE; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < CONV2_KENREL_SIZE; ++kx) {
                    int in_y = out_y + ky;
                    int in_x = out_x + kx;
                    float input_val = s_tile[curr_buf_idx][in_y][in_x];
                    #pragma unroll
                    for (int out_c = 0; out_c < CONV2_Co; ++out_c) {
                        float weight_val = s_weights[out_c][in_c][ky][kx];
                        acc[out_c] += input_val * weight_val;
                        // atomicAdd(&y[((size_t)batch * Co + out_c) * CONV2_Ho * CONV2_Wo + out_y * CONV2_Wo + out_x], input_val * weight_val);
                    }
                }
            }
        }
        // sync prefetch & compute
        __syncthreads();
    }
    // 3. epologue
    if (out_x < CONV2_Wo && out_y < CONV2_Ho){
        size_t y_base_offset = (size_t)batch * Co * CONV2_Ho * CONV2_Wo + out_y * CONV2_Wo + out_x;
        for (int out_c = 0; out_c < CONV2_Co; ++out_c) {
            float conv_output = acc[out_c] + s_bias[out_c];
            size_t y_channel_offset = y_base_offset + (size_t)out_c * CONV2_Ho * CONV2_Wo;
            
            // float vm = v[y_channel_offset];
            float vm = s_v[out_c][out_y][out_x];
            vm += conv_output;
            float spike = (vm >= 1.0f) ? 1.0f : 0.0f;
            
            y[y_channel_offset] = spike;
            // y[y_channel_offset] = conv_output;
            v[y_channel_offset] = vm * (1.0f - spike);

        }
    }
}
