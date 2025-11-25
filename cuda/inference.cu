#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
// Permitted CUDA headers for custom kernels
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define USE_WMMA 1
#else
#define USE_WMMA 0
#endif
#include <cstring>
#include <cuda.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#if USE_WMMA
using namespace nvcuda;
#endif

// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY BEGIN
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}
// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY BEGIN
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    std::vector<unsigned char> buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
        }
    }
    return images;
}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_items, 4); num_items = __builtin_bswap32(num_items);
    std::vector<int> labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for(int i = 0; i < num_items; ++i) { labels[i] = static_cast<int>(buffer[i]); }
    return labels;
}

std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params; float param;
    while (file >> param) { params.push_back(param); }
    return params;
}

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY END
// ===================================================================================


// clang-format off
#define BATCH_SIZE          512
#define STREAM_N            ((10000 + BATCH_SIZE - 1) / BATCH_SIZE)
#define NEURON_T            2
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define CONV_KERNEL_K       5

#define CONV2_KENREL_SIZE 5
#define CONV2_Hi 12
#define CONV2_Wi 12
#define CONV2_Ci 8
#define CONV2_IN_SIZE  (CONV2_Hi * CONV2_Wi)
#define CONV2_Ho (CONV2_Hi - CONV2_KENREL_SIZE + 1)
#define CONV2_Wo (CONV2_Wi - CONV2_KENREL_SIZE + 1)
#define CONV2_Co 16

// SNN-specific parameter, must match training
constexpr int TT = 2;

// Model fixed shapes from train_9004.py
constexpr int C1_IN = 1, C1_OUT = 8, K = 5;
constexpr int C2_IN = 8, C2_OUT = 16;
constexpr int I_H = 28, I_W = 28;
constexpr int C1_HO = I_H - K + 1;            // 24
constexpr int C1_WO = I_W - K + 1;            // 24
constexpr int P1_HO = C1_HO / 2;              // 12
constexpr int P1_WO = C1_WO / 2;              // 12
constexpr int C2_HO = P1_HO - K + 1;          // 8
constexpr int C2_WO = P1_WO - K + 1;          // 8
constexpr int P2_HO = C2_HO / 2;              // 4
constexpr int P2_WO = C2_WO / 2;              // 4
constexpr int FLAT  = C2_OUT * P2_HO * P2_WO; // 16*4*4=256
constexpr int FC1_O = 128;
constexpr int FC2_O = 96;
constexpr int FC3_O = 10;

constexpr int MAX_N = BATCH_SIZE;
// clang-format on

// ===================================================================================
// CUDA kernels and helpers for SCNN inference
// ===================================================================================

static inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

static void dump_host_f32(const char* path, const float* h, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f)
        return;
    fwrite(h, sizeof(float), n, f);
    fclose(f);
}


// conv general (NCHW), K=5 templated by compile-time constant
__global__ void conv2d_nchw_fuse_if(
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


__global__ void conv2d_c1_k5_fuse_if_kernel(
    const float* __restrict__ x, // [N, 1, 28, 28]
    const float* __restrict__ w, // [Co, 1, 5, 5]
    const float* __restrict__ b, // [Co]
    float* __restrict__ y,       // [N, Co, 24, 24]
    float* __restrict__ v,
    int N, int Co
) {
    constexpr int           K  = 5;
    const int               Hi = 28, Wi = 28;
    const int               Ho = 24, Wo = 24;
    extern __shared__ float tile[];
    const int               tileW = blockDim.x + K - 1;
    const int               tileH = blockDim.y + K - 1;
    const int               tx    = threadIdx.x;
    const int               ty    = threadIdx.y;
    const int               ow    = blockIdx.x * blockDim.x + tx;
    const int               oh    = blockIdx.y * blockDim.y + ty;
    const int               z     = blockIdx.z;
    const int               oc    = z % Co;
    const int               n     = z / Co;
    if (n >= N)
        return;
    const int ow0 = blockIdx.x * blockDim.x;
    const int oh0 = blockIdx.y * blockDim.y;
    for (int yy = ty; yy < tileH; yy += blockDim.y) {
        int ih = oh0 + yy;
        for (int xx = tx; xx < tileW; xx += blockDim.x) {
            int   iw = ow0 + xx;
            float v  = 0.0f;
            if (ih >= 0 && ih < Hi && iw >= 0 && iw < Wi) {
                const int x_idx = ((n * 1 + 0) * Hi + ih) * Wi + iw;
                v               = __ldg(x + x_idx);
            }
            tile[yy * tileW + xx] = v;
        }
    }
    __syncthreads();
    if (oh < Ho && ow < Wo) {
        const float* w_oc = w + oc * (K * K);
        float        acc  = __ldg(b + oc);
#pragma unroll
        for (int ky = 0; ky < K; ++ky) {
#pragma unroll
            for (int kx = 0; kx < K; ++kx) {
                float xv = tile[(ty + ky) * tileW + (tx + kx)];
                float ww = __ldg(w_oc + ky * K + kx);
                acc      = fmaf(xv, ww, acc);
            }
        }
        const int y_idx = ((n * Co + oc) * Ho + oh) * Wo + ow;
        float vm = v[y_idx] + acc;
        float spike = (vm >= 1.0f) ? 1.0f : 0.0f;
        y[y_idx]        = spike;
        v[y_idx] = vm * (1 - spike);
    }
}


__global__ void maxpool2x2_s2_nchw_kernel(
    const float* __restrict__ x, // [N, C, Hi, Wi]
    float* __restrict__ y,       // [N, C, Ho, Wo]
    int N, int C, int Hi, int Wi
) {
    const int Ho = Hi / 2;
    const int Wo = Wi / 2;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;
    const int z  = blockIdx.z; // n*c plane
    const int n  = z / C;
    const int c  = z % C;

    if (n >= N || c >= C || oh >= Ho || ow >= Wo)
        return;

    const int ih0    = oh * 2;
    const int iw0    = ow * 2;
    const int x_base = ((n * C + c) * Hi + ih0) * Wi + iw0;

    float v00 = __ldg(x + x_base);
    float v01 = (iw0 + 1 < Wi) ? __ldg(x + x_base + 1) : v00;
    float v10 = (ih0 + 1 < Hi) ? __ldg(x + x_base + Wi) : v00;
    float v11 = (ih0 + 1 < Hi && iw0 + 1 < Wi) ? __ldg(x + x_base + Wi + 1) : v00;

    float m0 = fmaxf(v00, v01);
    float m1 = fmaxf(v10, v11);
    float mv = fmaxf(m0, m1);

    const int y_idx = ((n * C + c) * Ho + oh) * Wo + ow;
    y[y_idx]        = mv;
}

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define MATMUL_WARP_M WMMA_M
#define MATMUL_WARP_N WMMA_N
#define MATMUL_WARP_K WMMA_K

#define NUM_MATMUL_WARPS_M 4
#define NUM_MATMUL_WARPS_N 2
#define NUM_MATMUL_WARPS_K 2

#define MATMUL_BLOCK_M (NUM_MATMUL_WARPS_M * MATMUL_WARP_M)
#define MATMUL_BLOCK_N (NUM_MATMUL_WARPS_N * MATMUL_WARP_N)
#define MATMUL_BLOCK_K (NUM_MATMUL_WARPS_K * MATMUL_WARP_K)


#define MATMUL_THREAD_M (NUM_MATMUL_WARPS_M * 32)
#define MATMUL_THREAD_N (NUM_MATMUL_WARPS_N)



// warp size = 4 * 2 , each warp compute a tile of size 64 * 32
#define MATMUL4_WARP_M WMMA_M
#define MATMUL4_WARP_N WMMA_N
#define MATMUL4_WARP_K WMMA_K

#define NUM_MATMUL4_WARPS_M 4
#define NUM_MATMUL4_WARPS_N 2
#define NUM_MATMUL4_WARPS_K 2

#define MATMUL4_BLOCK_M (NUM_MATMUL4_WARPS_M * MATMUL4_WARP_M)
#define MATMUL4_BLOCK_N (NUM_MATMUL4_WARPS_N * MATMUL4_WARP_N)
#define MATMUL4_BLOCK_K (NUM_MATMUL4_WARPS_K * MATMUL4_WARP_K)

#define MATMUL4_BLOCK_SKEW_M 16
#define MATMUL4_BLOCK_SKEW_N 16
#define MATMUL4_BLOCK_SKEW_K 16

#define MATMUL4_THREAD_M (NUM_MATMUL4_WARPS_M * 32)
#define MATMUL4_THREAD_N (NUM_MATMUL4_WARPS_N)

// Fully-connected forward: y = W x + b, W: [Out, In], x: [N, In], y: [N, Out]
// __global__ void fc_forward_kernel(
__global__ void linear_fuse_if(
    float* x, // [N, In]
    float* w, // [Out, In]
    float* b, // [Out]
    float*       y, // [N, Out]
    float*       v, // [N, Out]
    int M, int N, int K
) {
#if USE_WMMA
    const int BM      = MATMUL4_BLOCK_M;
    const int BN      = MATMUL4_BLOCK_N;
    const int BK      = MATMUL4_BLOCK_K;
    int       block_m = blockIdx.x * BM;
    int       block_n = blockIdx.y * BN;

    // prepare fragments
    nvcuda::wmma::
        fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>
            a_frag[NUM_MATMUL4_WARPS_M];
    nvcuda::wmma::
        fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major>
            b_frag[NUM_MATMUL4_WARPS_N];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        c_frag[NUM_MATMUL4_WARPS_M * NUM_MATMUL4_WARPS_N];
    // 缓存A_tile和B_tile
    __shared__ half As[MATMUL4_BLOCK_M * MATMUL4_BLOCK_K];
    __shared__ half Bs[MATMUL4_BLOCK_N * MATMUL4_BLOCK_K];
    __shared__ half Cs[MATMUL4_BLOCK_M * MATMUL4_BLOCK_N];

#pragma unroll
    for (int i = 0; i < NUM_MATMUL4_WARPS_M * NUM_MATMUL4_WARPS_N; i++) {
        nvcuda::wmma::fill_fragment(c_frag[i], __float2half(0.0f));
    }

#pragma unroll
    for (int block_k = 0; block_k < K; block_k += BK) {
        int threadid = (threadIdx.y * blockDim.x + threadIdx.x);
        // load x tile
        // float4 tmp = FETCH_FLOAT4(&A[]);
        for (int i = 0; i < BM * BK / 4; i += blockDim.x * blockDim.y ) {
            int index = threadid + i;
            int a_row = (index * 4) / BK;
            int a_col = (index * 4) % BK;
            if (index < BM * BK / 4) {
                float4 tmp = FETCH_FLOAT4(x[(block_m + a_row) * K + block_k + a_col]);
                half2 h2_0 = __float22half2_rn(make_float2(tmp.x, tmp.y));
                half2 h2_1 = __float22half2_rn(make_float2(tmp.z, tmp.w));
                *((half2*)&As[a_row * BK + a_col]) = h2_0;
                *((half2*)&As[a_row * BK + a_col+ 2]) = h2_1;
            }
        }

        __syncthreads();
        // load w tile
        for (int i = 0; i < BN * BK / 4; i += blockDim.x * blockDim.y) {
            int index = threadid + i;
            int b_row = (index * 4) / BK;
            int b_col = (index * 4) % BK;
            if (index < BN * BK / 4) {
                float4 tmp = FETCH_FLOAT4(w[(block_n + b_row) * K + block_k + b_col]);
                half2 h2_0 = __float22half2_rn(make_float2(tmp.x, tmp.y));
                half2 h2_1 = __float22half2_rn(make_float2(tmp.z, tmp.w));
                *((half2*)&Bs[b_row * BK + b_col]) = h2_0;
                *((half2*)&Bs[b_row * BK + b_col+ 2]) = h2_1;
            }
        }
        __syncthreads();


// calculate warp level
#pragma unroll
        for (int warp_k = 0; warp_k < BK; warp_k += MATMUL4_WARP_K) {
            // use wmma to compute y tile
            // flatten out 2d grid of threads into in order of increasing threadIdx.x
            int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;

            int warp_m = warp_id / NUM_MATMUL4_WARPS_N;
            int warp_n = warp_id % NUM_MATMUL4_WARPS_N;
            // printf("warp_id=%d, warp_m=%d, warp_n=%d\n", warp_id, warp_m, warp_n);


            // load x fragment
            int a_row = warp_m * MATMUL4_WARP_M;
            int a_col = warp_k;
            wmma::load_matrix_sync(a_frag[warp_m], &As[a_row * BK + a_col], BK);
            // load w fragment
            int b_row = warp_n * MATMUL4_WARP_N;
            int b_col = warp_k;
            wmma::load_matrix_sync(b_frag[warp_n], &Bs[b_row * BK + b_col], BK);
            // wmma mma
            wmma::mma_sync(c_frag[warp_id], a_frag[warp_m], b_frag[warp_n], c_frag[warp_id]);
            // store y fragment
        }
        __syncthreads();
    }
    // store y fragments to y

    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;
    int warp_m  = warp_id / NUM_MATMUL4_WARPS_N;
    int warp_n  = warp_id % NUM_MATMUL4_WARPS_N;
    int c_row   = block_m + warp_m * MATMUL4_WARP_M;
    int c_col   = block_n + warp_n * MATMUL4_WARP_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(
            &Cs[warp_m * MATMUL4_WARP_M * BN + warp_n * MATMUL4_WARP_N],
            c_frag[warp_id],
            BN,
            wmma::mem_row_major
        );
    }
    __syncthreads();
    // store y fragments to y
    int threadid = (threadIdx.y * blockDim.x + threadIdx.x);
    #pragma unroll
    for (int i = threadid; i < BM * BN / 4; i += blockDim.x * blockDim.y ) {
        int index = i;
        int c_row = (index * 4) / BN;
        int c_col = (index * 4) % BN;
        int gc_row = block_m + c_row;
        int gc_col = block_n + c_col;
        // printf("threadid=%d, i=%d, c_row=%d, c_col=%d, gc_row=%d, gc_col=%d\n", threadid, i, c_row, c_col, gc_row, gc_col);
        if (gc_row < M && gc_col + 3 < N) {
            float4 b4 = FETCH_FLOAT4(b[gc_col]);
            float4 v4 = FETCH_FLOAT4(v[gc_row * N + gc_col]);
            v4.x += __half2float(Cs[c_row * BN + c_col]) + b4.x;
            v4.y += __half2float(Cs[c_row * BN + c_col + 1]) + b4.y;
            v4.z += __half2float(Cs[c_row * BN + c_col + 2]) + b4.z;
            v4.w += __half2float(Cs[c_row * BN + c_col + 3]) + b4.w;
            float4 s;
            s.x = (v4.x >= 1.0f)? 1.0f : 0.0f;
            s.y = (v4.y >= 1.0f)? 1.0f : 0.0f;
            s.z = (v4.z >= 1.0f)? 1.0f : 0.0f;
            s.w = (v4.w >= 1.0f)? 1.0f : 0.0f;
            v4.x *= (1. - s.x);
            v4.y *= (1. - s.y);
            v4.z *= (1. - s.z);
            v4.w *= (1. - s.w);
            // FETCH_FLOAT4(v[gc_row * N + gc_col]) = s;
            float4* addr_v = (float4 *)&v[gc_row * N + gc_col];
            addr_v[0] = v4;
            // v[gc_row * N + gc_col] = v4.x;
            // v[gc_row * N + gc_col + 1] = v4.y;
            // v[gc_row * N + gc_col + 2] = v4.z;
            // v[gc_row * N + gc_col + 3] = v4.w;

            // float4 tmp = {__half2float(,
            //             __half2float(Cs[c_row * BN + c_col + 1]) + b4.y,
            //             __half2float(Cs[c_row * BN + c_col + 2]) + b4.z,
            //             __half2float(Cs[c_row * BN + c_col + 3]) + b4.w};
            float4 * addr_y = (float4 *)&y[gc_row * N + gc_col];
            addr_y[0] = s;
            // FETCH_FLOAT4(y[(gc_row) * N + gc_col]) = s;
            // y[gc_row * N + gc_col] = s.x;
            // y[gc_row * N + gc_col + 1] = s.y;
            // y[gc_row * N + gc_col + 2] = s.z;
            // y[gc_row * N + gc_col + 3] = s.w;

        }
    }

#endif
}


__global__ void linear_forward(
    float* x, // [N, In]
    float* w, // [Out, In]
    float* b, // [Out]
    float*       y, // [N, Out]
    int M, int N, int K
) {
#if USE_WMMA
    const int BM      = MATMUL4_BLOCK_M;
    const int BN      = MATMUL4_BLOCK_N;
    const int BK      = MATMUL4_BLOCK_K;
    int       block_m = blockIdx.x * BM;
    int       block_n = blockIdx.y * BN;

    // prepare fragments
    nvcuda::wmma::
        fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>
            a_frag[NUM_MATMUL4_WARPS_M];
    nvcuda::wmma::
        fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major>
            b_frag[NUM_MATMUL4_WARPS_N];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        c_frag[NUM_MATMUL4_WARPS_M * NUM_MATMUL4_WARPS_N];
    // 缓存A_tile和B_tile
    __shared__ half As[MATMUL4_BLOCK_M * MATMUL4_BLOCK_K];
    __shared__ half Bs[MATMUL4_BLOCK_N * MATMUL4_BLOCK_K];
    __shared__ half Cs[MATMUL4_BLOCK_M * MATMUL4_BLOCK_N];

#pragma unroll
    for (int i = 0; i < NUM_MATMUL4_WARPS_M * NUM_MATMUL4_WARPS_N; i++) {
        nvcuda::wmma::fill_fragment(c_frag[i], __float2half(0.0f));
    }

#pragma unroll
    for (int block_k = 0; block_k < K; block_k += BK) {
        int threadid = (threadIdx.y * blockDim.x + threadIdx.x);
        // load x tile
        // float4 tmp = FETCH_FLOAT4(&A[]);
        for (int i = 0; i < BM * BK / 4; i += blockDim.x * blockDim.y ) {
            int index = threadid + i;
            int a_row = (index * 4) / BK;
            int a_col = (index * 4) % BK;
            if (index < BM * BK / 4) {
                float4 tmp = FETCH_FLOAT4(x[(block_m + a_row) * K + block_k + a_col]);
                half2 h2_0 = __float22half2_rn(make_float2(tmp.x, tmp.y));
                half2 h2_1 = __float22half2_rn(make_float2(tmp.z, tmp.w));
                *((half2*)&As[a_row * BK + a_col]) = h2_0;
                *((half2*)&As[a_row * BK + a_col+ 2]) = h2_1;
            }
        }

        // for (int i = threadIdx.y; i < BM; i += blockDim.y) {
        //     for (int j = threadIdx.x; j < BK; j += blockDim.x) {
        //         int a_row = block_m + i;
        //         int a_col = block_k + j;
        //         if (a_row < M && a_col < K) {
        //             As[i * BK + j] = __float2half(A[a_row * K + a_col]);
        //         }
        //     }
        // }
        __syncthreads();
        // load w tile
        for (int i = 0; i < BN * BK / 4; i += blockDim.x * blockDim.y) {
            int index = threadid + i;
            int b_row = (index * 4) / BK;
            int b_col = (index * 4) % BK;
            if (index < BN * BK / 4) {
                float4 tmp = FETCH_FLOAT4(w[(block_n + b_row) * K + block_k + b_col]);
                half2 h2_0 = __float22half2_rn(make_float2(tmp.x, tmp.y));
                half2 h2_1 = __float22half2_rn(make_float2(tmp.z, tmp.w));
                *((half2*)&Bs[b_row * BK + b_col]) = h2_0;
                *((half2*)&Bs[b_row * BK + b_col+ 2]) = h2_1;
            }
        }
        // for (int i = threadIdx.y; i < BN; i += blockDim.y) {
        //     for (int j = threadIdx.x; j < BK; j += blockDim.x) {
        //         int b_row = block_n + i;
        //         int b_col = block_k + j;
        //         if (b_row < N && b_col < K) {
        //             Bs[i * BK + j] = __float2half(B[b_row * K + b_col]);
        //         }
        //     }
        // }
        __syncthreads();


// calculate warp level
#pragma unroll
        for (int warp_k = 0; warp_k < BK; warp_k += MATMUL4_WARP_K) {
            // use wmma to compute y tile
            // flatten out 2d grid of threads into in order of increasing threadIdx.x
            int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;

            int warp_m = warp_id / NUM_MATMUL4_WARPS_N;
            int warp_n = warp_id % NUM_MATMUL4_WARPS_N;
            // printf("warp_id=%d, warp_m=%d, warp_n=%d\n", warp_id, warp_m, warp_n);


            // load x fragment
            int a_row = warp_m * MATMUL4_WARP_M;
            int a_col = warp_k;
            wmma::load_matrix_sync(a_frag[warp_m], &As[a_row * BK + a_col], BK);
            // load w fragment
            int b_row = warp_n * MATMUL4_WARP_N;
            int b_col = warp_k;
            wmma::load_matrix_sync(b_frag[warp_n], &Bs[b_row * BK + b_col], BK);
            // wmma mma
            wmma::mma_sync(c_frag[warp_id], a_frag[warp_m], b_frag[warp_n], c_frag[warp_id]);
            // store y fragment
        }
        __syncthreads();
    }
    // store y fragments to y

    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / warpSize;
    int warp_m  = warp_id / NUM_MATMUL4_WARPS_N;
    int warp_n  = warp_id % NUM_MATMUL4_WARPS_N;
    int c_row   = block_m + warp_m * MATMUL4_WARP_M;
    int c_col   = block_n + warp_n * MATMUL4_WARP_N;
    if (c_row < M && c_col < N) {
        wmma::store_matrix_sync(
            &Cs[warp_m * MATMUL4_WARP_M * BN + warp_n * MATMUL4_WARP_N],
            c_frag[warp_id],
            BN,
            wmma::mem_row_major
        );
    }
    __syncthreads();
#pragma unroll
    for (int i = threadIdx.y; i < BM; i += blockDim.y) {
        for (int j = threadIdx.x; j < BN; j += blockDim.x) {
            int c_row = block_m + i;
            int c_col = block_n + j;
            if (c_row < M && c_col < N) {
                Cs[i * BN + j]       = __float2half(__half2float(Cs[i * BN + j]) + b[c_col]);
                y[c_row * N + c_col] = __half2float(Cs[i * BN + j]);
            }
        }
    }
#endif
}

__global__ void add_inplace_kernel(float* __restrict__ a, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    a[i] += b[i];
}

__global__ void scale_inplace_kernel(float* __restrict__ a, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    a[i] *= s;
}

__global__ void argmax10_kernel(const float* __restrict__ logits, int* __restrict__ preds, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N)
        return;
    const float* row    = logits + n * 10;
    int          best_k = 0;
    float        best_v = row[0];
    for (int k = 1; k < 10; ++k) {
        float v = row[k];
        if (v > best_v) {
            best_v = v;
            best_k = k;
        }
    }
    preds[n] = best_k;
}

struct Global {
    static Global& Get()
    {
        static Global instance;
        return instance;
    }

    ~Global()
    {

        // Free batch buffers
        for (int i = 0; i < STREAM_N; i ++)
        {
            checkCudaErrors(cudaStreamDestroy(stream[i]));
            checkCudaErrors(cudaFree(d_workspace[i]));
            checkCudaErrors(cudaFreeHost(h_batch[i]));
            checkCudaErrors(cudaFreeHost(h_batch_preds[i]));
        }
    }

    constexpr static size_t d_input_size = MAX_N * C1_IN * I_H * I_W; 
    constexpr static size_t d_conv1_out_size = MAX_N * C1_OUT * C1_HO * C1_WO;
    constexpr static size_t d_if1_mem_size = d_conv1_out_size;
    // constexpr static size_t d_if1_spk_size = d_conv1_out_size;
    constexpr static size_t d_pool1_out_size = MAX_N * C1_OUT * P1_HO * P1_WO;
    constexpr static size_t d_conv2_out_size = MAX_N * C2_OUT * C2_HO * C2_WO;
    constexpr static size_t d_if2_mem_size = d_conv2_out_size;
    // constexpr static size_t d_if2_spk_size = d_conv2_out_size;
    constexpr static size_t d_pool2_out_size = MAX_N * C2_OUT * P2_HO * P2_WO;
    constexpr static size_t d_fc1_out_size = MAX_N * FC1_O;
    constexpr static size_t d_if3_mem_size = d_fc1_out_size;
    // constexpr static size_t d_if3_spk_size = d_fc1_out_size;
    constexpr static size_t d_fc2_out_size = MAX_N * FC2_O;
    constexpr static size_t d_if4_mem_size = d_fc2_out_size;
    // constexpr static size_t d_if4_spk_size = d_fc2_out_size;
    constexpr static size_t d_fc3_out_size = MAX_N * FC3_O;
    constexpr static size_t d_logits_sum_size = d_fc3_out_size;
    constexpr static size_t d_preds_size = MAX_N;

    size_t total_d_size = d_input_size + d_conv1_out_size + d_if1_mem_size + d_pool1_out_size +
                          d_conv2_out_size + d_if2_mem_size  +
                          d_pool2_out_size + d_fc1_out_size + d_if3_mem_size +
                          d_fc2_out_size + d_if4_mem_size +
                          d_fc3_out_size + d_logits_sum_size +
                          d_preds_size;

    size_t d_input_offset = 0;
    size_t d_conv1_out_offset = d_input_offset + d_input_size;
    size_t d_if1_mem_offset = d_conv1_out_offset + d_conv1_out_size;
    // size_t d_if1_spk_offset = d_if1_mem_offset + d_if1_mem_size;
    size_t d_pool1_out_offset = d_if1_mem_offset + d_if1_mem_size;
    size_t d_conv2_out_offset = d_pool1_out_offset + d_pool1_out_size;
    size_t d_if2_mem_offset = d_conv2_out_offset + d_conv2_out_size;
    // size_t d_if2_spk_offset = d_if2_mem_offset + d_if2_mem_size;
    size_t d_pool2_out_offset = d_if2_mem_offset + d_if2_mem_size;
    size_t d_fc1_out_offset = d_pool2_out_offset + d_pool2_out_size;
    size_t d_if3_mem_offset = d_fc1_out_offset + d_fc1_out_size;
    // size_t d_if3_spk_offset = d_if3_mem_offset + d_if3_mem_size;
    size_t d_fc2_out_offset = d_if3_mem_offset + d_if3_mem_size;
    size_t d_if4_mem_offset = d_fc2_out_offset + d_fc2_out_size;
    // size_t d_if4_spk_offset = d_if4_mem_offset + d_if4_mem_size;
    size_t d_fc3_out_offset = d_if4_mem_offset + d_if4_mem_size;
    size_t d_logits_sum_offset = d_fc3_out_offset + d_fc3_out_size;
    size_t d_preds_offset = d_logits_sum_offset + d_logits_sum_size;
    float * d_workspace[STREAM_N];
    float * h_batch[STREAM_N];
    int * h_batch_preds[STREAM_N];

    
    cudaStream_t stream[STREAM_N];


private:
    Global() {
        // Pre-allocate device buffers for the maximum batch size to reuse across batches
        for (int i = 0; i < STREAM_N; i ++)
        {
            checkCudaErrors(cudaStreamCreate(&stream[i]));
            checkCudaErrors(cudaMalloc(&d_workspace[i], total_d_size * sizeof(float)));
            checkCudaErrors(cudaMallocHost(&h_batch[i], sizeof(float) * MAX_N * C1_IN * I_H * I_W));
            checkCudaErrors(cudaMallocHost(&h_batch_preds[i], sizeof(int) * MAX_N));
        }
    }
};

auto& global = Global::Get();

std::vector<int> scnn_inference(
    const std::vector<std::vector<float>>& images,
    // Device pointers for parameters
    float* d_conv1_w, float* d_conv1_b, float* d_conv2_w, float* d_conv2_b, float* d_fc1_w,
    float* d_fc1_b, float* d_fc2_w, float* d_fc2_b, float* d_fc3_w, float* d_fc3_b
    // YOU CAN ADD MORE PARAMETERS HERE!!!
) {
    std::vector<int> predictions;
    const int        num_images = images.size(); // 1000
    predictions.resize(num_images);


    // printf("num images: %d\n", num_images);

    for (int offset = 0; offset < num_images; offset += MAX_N) {
        int N = std::min(MAX_N, num_images - offset);
        int stream_id = offset / MAX_N;
        auto& stream = global.stream[stream_id];

        float * d_input = global.d_workspace[stream_id] + global.d_input_offset;
        float * d_conv1_out = global.d_workspace[stream_id] + global.d_conv1_out_offset;
        float * d_if1_mem = global.d_workspace[stream_id] + global.d_if1_mem_offset;
        float * d_pool1_out = global.d_workspace[stream_id] + global.d_pool1_out_offset;
        float * d_conv2_out = global.d_workspace[stream_id] + global.d_conv2_out_offset;
        float * d_if2_mem = global.d_workspace[stream_id] + global.d_if2_mem_offset;
        float * d_pool2_out = global.d_workspace[stream_id] + global.d_pool2_out_offset;
        float * d_fc1_out = global.d_workspace[stream_id] + global.d_fc1_out_offset;
        float * d_if3_mem = global.d_workspace[stream_id] + global.d_if3_mem_offset;
        float * d_fc2_out = global.d_workspace[stream_id] + global.d_fc2_out_offset;
        float * d_if4_mem = global.d_workspace[stream_id] + global.d_if4_mem_offset;
        float * d_fc3_out = global.d_workspace[stream_id] + global.d_fc3_out_offset;
        float * d_logits_sum = global.d_workspace[stream_id] + global.d_logits_sum_offset;
        int   * d_preds = (int*)(global.d_workspace[stream_id] + global.d_preds_offset);

        // Pack host batch into contiguous buffer
        const int in_elems_per_img = C1_IN * I_H * I_W; // 784
        for (int n = 0; n < N; ++n) {
            const float* src = images[offset + n].data();
            float*       dst = global.h_batch[stream_id] + n * in_elems_per_img;
            
            std::memcpy(dst, src, in_elems_per_img * sizeof(float));
        }

        // Copy inputs to device
        checkCudaErrors(cudaMemcpyAsync(
            d_input,
            global.h_batch[stream_id],
            N * in_elems_per_img * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        ));

        // Zero IF memories and logits sum for this batch
        checkCudaErrors(cudaMemsetAsync(d_if1_mem, 0, N * C1_OUT * C1_HO * C1_WO * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(d_if2_mem, 0, N * C2_OUT * C2_HO * C2_WO * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(d_if3_mem, 0, N * FC1_O * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(d_if4_mem, 0, N * FC2_O * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(d_logits_sum, 0, N * FC3_O * sizeof(float), stream));

        // Common launch configs
        const dim3 block2d(16, 16, 1);
        // Conv1: (N*Co) in grid.z, K=5 shared mem tile size
        const size_t conv1_smem = (block2d.x + K - 1) * (block2d.y + K - 1) * sizeof(float);
        const dim3   grid_c1(div_up(C1_WO, block2d.x), div_up(C1_HO, block2d.y), N * C1_OUT);
        const dim3   grid_c2(div_up(C2_WO, block2d.x), div_up(C2_HO, block2d.y), N);
        // Pool kernels configs computed per layer
        const dim3 grid_pool1(div_up(P1_WO, block2d.x), div_up(P1_HO, block2d.y), N * C1_OUT);
        const dim3 grid_pool2(div_up(P2_WO, block2d.x), div_up(P2_HO, block2d.y), N * C2_OUT);

        // IF launch config
        const int threads1d = 256;
        // static int cnt = 0;
        for (int t = 0; t < TT; ++t) {
            // printf("run on %d", ++cnt);
            // Conv1: Cin=1 fast path
            conv2d_c1_k5_fuse_if_kernel<<<grid_c1, block2d, conv1_smem, stream>>>(
                d_input,
                d_conv1_w,
                d_conv1_b,
                d_conv1_out,
                d_if1_mem,
                N,
                C1_OUT
            );
            #if defined(CUDA_DEBUG)
                checkCudaErrors(cudaGetLastError());
            #endif

            // Pool1: 2x2 stride2
            maxpool2x2_s2_nchw_kernel<<<grid_pool1, block2d, 0, stream>>>(
                d_conv1_out,
                d_pool1_out,
                N,
                C1_OUT,
                C1_HO,
                C1_WO
            );
            #if defined(CUDA_DEBUG)
                checkCudaErrors(cudaGetLastError());
            #endif

            // Conv2: general K=5, grid.z = N only
            {
                const dim3 block2d_c2(8, 8, 1);
                conv2d_nchw_fuse_if<<<grid_c2, block2d_c2, 0, stream >>>(
                    d_pool1_out,
                    d_conv2_w,
                    d_conv2_b,
                    d_conv2_out,
                    d_if2_mem,
                    N,
                    C2_IN,
                    P1_HO,
                    P1_WO,
                    C2_OUT
                );
                #if defined(CUDA_DEBUG)
                    checkCudaErrors(cudaGetLastError());
                #endif
            }

            // Pool2
            maxpool2x2_s2_nchw_kernel<<<grid_pool2, block2d,0, stream>>>(
                d_conv2_out,
                d_pool2_out,
                N,
                C2_OUT,
                C2_HO,
                C2_WO
            );
            #if defined(CUDA_DEBUG)
                checkCudaErrors(cudaGetLastError());
            #endif

            // Flatten is just reinterpretation: [N, 16, 4, 4] -> [N, 256]

            // FC1
            {
                const int  In = FLAT, Out = FC1_O;
                dim3 blockDim(MATMUL_THREAD_M, MATMUL_THREAD_N);
                dim3 gridDim(div_up(N, MATMUL_BLOCK_M), div_up(Out, MATMUL_BLOCK_N));
                // size_t     smem = In * sizeof(float);
                linear_fuse_if<<<gridDim, blockDim, 0, stream>>>(
                    d_pool2_out,
                    d_fc1_w,
                    d_fc1_b,
                    d_fc1_out,
                    d_if3_mem,
                    N,
                    Out,
                    In
                );
                #if defined(CUDA_DEBUG)
                    checkCudaErrors(cudaGetLastError());
                #endif
            }

            // FC2
            {
                const int  In = FC1_O, Out = FC2_O;
                dim3 blockDim(MATMUL_THREAD_M, MATMUL_THREAD_N);
                dim3 gridDim(div_up(N, MATMUL_BLOCK_M), div_up(Out, MATMUL_BLOCK_N));
                // const dim3 block(256, 1, 1);
                // const dim3 grid(div_up(Out, 256), N, 1);
                // size_t     smem = In * sizeof(float);
                linear_fuse_if<<<gridDim, blockDim, 0, stream>>>(
                    d_fc1_out,
                    d_fc2_w,
                    d_fc2_b,
                    d_fc2_out,
                    d_if4_mem,
                    N,
                    Out,
                    In
                );
                #if defined(CUDA_DEBUG)
                    checkCudaErrors(cudaGetLastError());
                #endif
            }

            // FC3 (logits for this timestep)
            {
                const int  In = FC2_O, Out = FC3_O;
                dim3 blockDim(MATMUL_THREAD_M, MATMUL_THREAD_N);
                dim3 gridDim(div_up(N, MATMUL_BLOCK_M), div_up(Out, MATMUL_BLOCK_N));
                // const dim3 block(256, 1, 1);
                // const dim3 grid(div_up(Out, 256), N, 1);
                // size_t     smem = In * sizeof(float);
                linear_forward<<<gridDim, blockDim, 0, stream>>>(
                    d_fc2_out,
                    d_fc3_w,
                    d_fc3_b,
                    d_fc3_out,
                    N,
                    Out,
                    In
                );
                #if defined(CUDA_DEBUG)
                    checkCudaErrors(cudaGetLastError());
                #endif
            }

            // Accumulate logits across time: logits_sum += d_fc3_out
            {
                int total  = N * FC3_O;
                int blocks = div_up(total, threads1d);
                add_inplace_kernel<<<blocks, threads1d,0, stream>>>(d_logits_sum, d_fc3_out, total);
                #if defined(CUDA_DEBUG)
                    checkCudaErrors(cudaGetLastError());
                #endif
            }
        }

        // Average logits over T
        {
            int total  = N * FC3_O;
            int blocks = div_up(total, threads1d);
            scale_inplace_kernel<<<blocks, threads1d,0, stream>>>(d_logits_sum, 1.0f / float(TT), total);
            #if defined(CUDA_DEBUG)
                checkCudaErrors(cudaGetLastError());
            #endif
        }

        // Argmax to predictions
        {
            const int threads = 256;
            const int blocks  = div_up(N, threads);
            argmax10_kernel<<<blocks, threads,0, stream>>>(d_logits_sum, d_preds, N);
            #if defined(CUDA_DEBUG)
                checkCudaErrors(cudaGetLastError());
            #endif
        }

                // Copy predictions to host and append
        checkCudaErrors(
            cudaMemcpyAsync(global.h_batch_preds[stream_id], d_preds, N * sizeof(int), cudaMemcpyDeviceToHost, stream)
        );

        // Copy predictions to host and append
        // checkCudaErrors(
        //     cudaMemcpyAsync(batch_preds.data(), d_preds, N * sizeof(int), cudaMemcpyDeviceToHost, stream)
        // );
        // for (int n = 0; n < N; ++n) {
        //     predictions.push_back(batch_preds[n]);
        // }
    }

    for (int offset = 0; offset < num_images; offset += MAX_N) {
        int N = std::min(MAX_N, num_images - offset);
        int stream_id = offset / MAX_N;
        // Wait for stream to finish
        checkCudaErrors(cudaStreamSynchronize(global.stream[stream_id]));
        // Append predictions
        memcpy(predictions.data() + offset, global.h_batch_preds[stream_id], N * sizeof(int));
    }

    // Memory is freed in main for parameters.
    return predictions;
}


// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_and_data_dir>" << std::endl;
        return 1;
    }
	std::string dir = argv[1];
	
    // Load test data
    auto images = read_mnist_images(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    if (images.empty() || labels.empty()) return 1;

    // Load model parameters to host memory
    auto conv1_w = read_param(dir + "/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/conv2.bias.txt");
    auto fc1_w = read_param(dir + "/fc1.weight.txt");
    auto fc1_b = read_param(dir + "/fc1.bias.txt");
    auto fc2_w = read_param(dir + "/fc2.weight.txt");
    auto fc2_b = read_param(dir + "/fc2.bias.txt");
    auto fc3_w = read_param(dir + "/fc3.weight.txt");
    auto fc3_b = read_param(dir + "/fc3.bias.txt");
    
    // --- 1. Allocate all necessary GPU memory ---
    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters
    checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_w,   fc1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_b,   fc1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_w,   fc2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_b,   fc2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_w,   fc3_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_b,   fc3_b.size() * sizeof(float)));

    // --- 2. Copy constant parameters from host to device ---
    checkCudaErrors(cudaMemcpy(d_conv1_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_w, fc3_w.data(), fc3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_b, fc3_b.data(), fc3_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================

    // --- 3. Perform inference ---
    // Pass device pointers to the inference function
    std::vector<int> predictions = scnn_inference(images,
        d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
        d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b
        // YOU CAN ADD MORE PARAMETERS HERE!!!
        );
    
// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================

    // Synchronize to ensure all GPU work is done before stopping the timer
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // --- 4. Free all allocated GPU memory ---
    checkCudaErrors(cudaFree(d_conv1_w));
    checkCudaErrors(cudaFree(d_conv1_b));
    checkCudaErrors(cudaFree(d_conv2_w));
    checkCudaErrors(cudaFree(d_conv2_b));
    checkCudaErrors(cudaFree(d_fc1_w));
    checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));
    checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));
    checkCudaErrors(cudaFree(d_fc3_b));
    
    // Calculate accuracy
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();
    
    // Output result in the required format
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;
    
    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================