#include "kernel.cuh"
#include "common.cuh"
#include <cuda_runtime.h>



// conv general (NCHW), K=5 templated by compile-time constant
__global__ void conv2d_nchw_kernel_n_only(
    const float* __restrict__ x, // [N, Ci, Hi, Wi]
    const float* __restrict__ w, // [Co, Ci, K, K]
    const float* __restrict__ b, // [Co]
    float* __restrict__ y,       // [N, Co, Ho, Wo]
    int N, int Ci, int Hi, int Wi, int Co
) {
    const int Ho = Hi - CONV_KERNEL_K + 1;
    const int Wo = Wi - CONV_KERNEL_K + 1;
    const int n  = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;
    if (n >= N || oh >= Ho || ow >= Wo)
        return;
    for (int oc = 0; oc < Co; ++oc) {
        float acc = __ldg(b + oc);
        for (int ci = 0; ci < Ci; ++ci) {
            const int x_base = ((n * Ci + ci) * Hi + oh) * Wi + ow;
            const int w_base = ((oc * Ci + ci) * CONV_KERNEL_K) * CONV_KERNEL_K;
#pragma unroll
            for (int ky = 0; ky < CONV_KERNEL_K; ++ky) {
#pragma unroll
                for (int kx = 0; kx < CONV_KERNEL_K; ++kx) {
                    float xv = __ldg(x + x_base + ky * Wi + kx);
                    float ww = __ldg(w + w_base + ky * CONV_KERNEL_K + kx);
                    acc      = fmaf(xv, ww, acc);
                }
            }
        }
        const int y_idx = ((n * Co + oc) * Ho + oh) * Wo + ow;
        y[y_idx]        = acc;
    }
}

// conv Cin=1 K=5 fast path with shared memory
__global__ void conv2d_c1_k5_kernel(
    const float* __restrict__ x, // [N, 1, 28, 28]
    const float* __restrict__ w, // [Co, 1, 5, 5]
    const float* __restrict__ b, // [Co]
    float* __restrict__ y,       // [N, Co, 24, 24]
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
        y[y_idx]        = acc;
    }
}

// conv general (NCHW), K=5 templated by compile-time constant
__global__ void conv2d_nchw_fuse_if_kernel_n_only(
    const float* __restrict__ x, // [N, Ci, Hi, Wi]
    const float* __restrict__ w, // [Co, Ci, K, K]
    const float* __restrict__ b, // [Co]
    float* __restrict__ y,       // [N, Co, Ho, Wo]
    float* __restrict__ v,
    int N, int Ci, int Hi, int Wi, int Co
) {
    const int Ho = Hi - CONV_KERNEL_K + 1;
    const int Wo = Wi - CONV_KERNEL_K + 1;
    const int n  = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;
    if (n >= N || oh >= Ho || ow >= Wo)
        return;
    for (int oc = 0; oc < Co; ++oc) {
        float acc = __ldg(b + oc);
        for (int ci = 0; ci < Ci; ++ci) {
            const int x_base = ((n * Ci + ci) * Hi + oh) * Wi + ow;
            const int w_base = ((oc * Ci + ci) * CONV_KERNEL_K) * CONV_KERNEL_K;
#pragma unroll
            for (int ky = 0; ky < CONV_KERNEL_K; ++ky) {
#pragma unroll
                for (int kx = 0; kx < CONV_KERNEL_K; ++kx) {
                    float xv = __ldg(x + x_base + ky * Wi + kx);
                    float ww = __ldg(w + w_base + ky * CONV_KERNEL_K + kx);
                    acc      = fmaf(xv, ww, acc);
                }
            }
        }
        const int y_idx = ((n * Co + oc) * Ho + oh) * Wo + ow;
        float vm = v[y_idx] + acc;
        float spike = (vm >= 1.0f) ? 1.0f : 0.0f;
        y[y_idx]        = spike;
        v[y_idx] = vm * (1 - spike);
        // y[y_idx]        = acc;
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

    // 使用向量化加载：将 tile 展平后按 float4 加载
    const int numThreads = blockDim.x * blockDim.y;
    const int threadId   = ty * blockDim.x + tx;
    const int x_base     = n * Hi * Wi; // 输入图像基地址 (单通道)
    const int tileSize   = tileW * tileH;

    // 每个线程负责加载多个 float4
    // 按照线性索引遍历 tile，每次加载 4 个连续元素
    for (int base_idx = threadId * 4; base_idx < tileSize; base_idx += numThreads * 4) {
        int row0 = base_idx / tileW;
        int col0 = base_idx % tileW;

        // 检查这 4 个元素是否在同一行内（避免跨行访问）
        bool sameRow = (col0 + 3 < tileW);

        if (sameRow) {
            int ih = oh0 + row0;
            int iw = ow0 + col0;

            // 检查是否在有效输入范围内，且全局内存地址对齐
            bool inBounds = (ih >= 0 && ih < Hi && iw >= 0 && iw + 3 < Wi);
            bool aligned  = ((size_t)(x + x_base + ih * Wi + iw) & 0xF) == 0;

            if (inBounds && aligned) {
                // 使用 float4 向量化加载和存储
                float4 data = *reinterpret_cast<const float4*>(x + x_base + ih * Wi + iw);
                FETCH_FLOAT4(tile[base_idx]) = data;
            } else {
                // 边界或未对齐情况：逐个加载
                #pragma unroll
                for (int j = 0; j < 4 && base_idx + j < tileSize; ++j) {
                    int cur_iw = iw + j;
                    float val = 0.0f;
                    if (ih >= 0 && ih < Hi && cur_iw >= 0 && cur_iw < Wi) {
                        val = __ldg(x + x_base + ih * Wi + cur_iw);
                    }
                    tile[base_idx + j] = val;
                }
            }
        } else {
            // 跨行情况：逐个加载
            #pragma unroll
            for (int j = 0; j < 4 && base_idx + j < tileSize; ++j) {
                int idx = base_idx + j;
                int row = idx / tileW;
                int col = idx % tileW;
                int ih = oh0 + row;
                int iw = ow0 + col;
                float val = 0.0f;
                if (ih >= 0 && ih < Hi && iw >= 0 && iw < Wi) {
                    val = __ldg(x + x_base + ih * Wi + iw);
                }
                tile[idx] = val;
            }
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


__global__ void ifnode_integrate_fire_kernel(
    const float* __restrict__ x, float* __restrict__ mem, float* __restrict__ spk, float thr,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    float v = mem[i] + x[i];
    float s = v >= thr ? 1.0f : 0.0f;
    spk[i]  = s;
    mem[i]  = (s > 0.f) ? 0.f : v;
}

__global__ void maxpool2x2_s2_nchw_kernel(
    const float* __restrict__ x, float* __restrict__ y, int N, int C, int Hi, int Wi
) {
    const int Ho = Hi / 2;
    const int Wo = Wi / 2;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int ow = blockIdx.x * blockDim.x + tx;
    const int oh = blockIdx.y * blockDim.y + ty;
    const int z  = blockIdx.z;
    const int n  = z / C;
    const int c  = z % C;
    if (n >= N || c >= C || oh >= Ho || ow >= Wo)
        return;
    const int ih0    = oh * 2;
    const int iw0    = ow * 2;
    const int x_base = ((n * C + c) * Hi + ih0) * Wi + iw0;
    float     v00    = __ldg(x + x_base);
    float     v01    = (iw0 + 1 < Wi) ? __ldg(x + x_base + 1) : v00;
    float     v10    = (ih0 + 1 < Hi) ? __ldg(x + x_base + Wi) : v00;
    float     v11    = (ih0 + 1 < Hi && iw0 + 1 < Wi) ? __ldg(x + x_base + Wi + 1) : v00;
    float     m0     = fmaxf(v00, v01);
    float     m1     = fmaxf(v10, v11);
    float     mv     = fmaxf(m0, m1);
    const int y_idx  = ((n * C + c) * Ho + oh) * Wo + ow;
    y[y_idx]         = mv;
}

__global__ void fc_forward_kernel(
    const float* __restrict__ x, // [N, In]
    const float* __restrict__ W, // [Out, In]
    const float* __restrict__ b, // [Out]
    float* __restrict__ y,       // [N, Out]
    int N, int In, int Out
) {
    extern __shared__ float xs[]; // size In
    const int               n = blockIdx.y;
    if (n >= N)
        return;
    for (int i = threadIdx.x; i < In; i += blockDim.x) {
        xs[i] = __ldg(x + n * In + i);
    }
    __syncthreads();
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= Out)
        return;
    const float* w_row = W + out_idx * In;
    float        acc   = __ldg(b + out_idx);
    for (int i = 0; i < In; ++i) {
        acc = fmaf(xs[i], __ldg(w_row + i), acc);
    }
    y[n * Out + out_idx] = acc;
}
