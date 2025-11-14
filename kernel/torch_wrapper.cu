#include <torch/extension.h>
#include "kernel.cuh"
#include "linear.cuh"


static inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}


#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)


// ------------------ Wrappers ------------------


void conv1_c1_k5_implicit(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda() && y.is_cuda(), "CUDA tensors req");
    TORCH_CHECK(
        x.dtype() == torch::kFloat32 && w.dtype() == torch::kFloat32 &&
            b.dtype() == torch::kFloat32 && y.dtype() == torch::kFloat32,
        "float32 only"
    );
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4 && y.dim() == 4, "NCHW shapes");
    const int  N  = x.size(0);
    const int  Co = w.size(0);
    const dim3 block(16, 16, 1);
    const dim3 grid(div_up(24, block.x), div_up(24, block.y), N * Co);
    // size_t     smem = (block.x + 4) * (block.y + 4) * sizeof(float);
    conv2d_c1_k5_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        N,
        Co
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1 kernel failed: ", cudaGetErrorString(err));
}

void conv1_c1k5(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda() && y.is_cuda(), "CUDA tensors req");
    TORCH_CHECK(
        x.dtype() == torch::kFloat32 && w.dtype() == torch::kFloat32 &&
            b.dtype() == torch::kFloat32 && y.dtype() == torch::kFloat32,
        "float32 only"
    );
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4 && y.dim() == 4, "NCHW shapes");
    const int  N  = x.size(0);
    const int  Co = w.size(0);
    const dim3 block(16, 16, 1);
    const dim3 grid(div_up(24, block.x), div_up(24, block.y), N * Co);
    size_t     smem = (block.x + 4) * (block.y + 4) * sizeof(float);
    conv2d_c1_k5_kernel<<<grid, block, smem>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        N,
        Co
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv1 kernel failed: ", cudaGetErrorString(err));
}

void conv2_k5(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && b.is_cuda() && y.is_cuda(), "CUDA tensors req");
    TORCH_CHECK(
        x.dtype() == torch::kFloat32 && w.dtype() == torch::kFloat32 &&
            b.dtype() == torch::kFloat32 && y.dtype() == torch::kFloat32,
        "float32 only"
    );
    TORCH_CHECK(x.dim() == 4 && w.dim() == 4 && y.dim() == 4, "NCHW shapes");
    const int  N = x.size(0), Ci = x.size(1), Hi = x.size(2), Wi = x.size(3), Co = w.size(0);
    const int  Ho = Hi - 4, Wo = Wi - 4;
    const dim3 block(16, 16, 1);
    const dim3 grid(div_up(Wo, block.x), div_up(Ho, block.y), N);
    conv2d_nchw_kernel_n_only<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        N,
        Ci,
        Hi,
        Wi,
        Co
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv2 kernel failed: ", cudaGetErrorString(err));
}

void pool2x2s2(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "CUDA tensors req");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && y.dtype() == torch::kFloat32, "float32 only");
    const int N = x.size(0), C = x.size(1), Hi = x.size(2), Wi = x.size(3);
    const int Ho = Hi / 2, Wo = Wi / 2;
    (void)Ho;
    (void)Wo;
    const dim3 block(16, 16, 1);
    const dim3 grid(div_up(Wo, block.x), div_up(Ho, block.y), N * C);
    maxpool2x2_s2_nchw_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N,
        C,
        Hi,
        Wi
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "pool kernel failed: ", cudaGetErrorString(err));
}

void if_integrate(torch::Tensor x, torch::Tensor mem, torch::Tensor spk, double thr) {
    TORCH_CHECK(x.is_cuda() && mem.is_cuda() && spk.is_cuda(), "CUDA tensors req");
    TORCH_CHECK(
        x.dtype() == torch::kFloat32 && mem.dtype() == torch::kFloat32 &&
            spk.dtype() == torch::kFloat32,
        "float32 only"
    );
    const int total   = x.numel();
    const int threads = 256;
    const int blocks  = div_up(total, threads);
    ifnode_integrate_fire_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        mem.data_ptr<float>(),
        spk.data_ptr<float>(),
        static_cast<float>(thr),
        total
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "if kernel failed: ", cudaGetErrorString(err));
}

void fc_forward(torch::Tensor x, torch::Tensor W, torch::Tensor b, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda() && b.is_cuda() && y.is_cuda(), "CUDA tensors req");
    TORCH_CHECK(
        x.dtype() == torch::kFloat32 && W.dtype() == torch::kFloat32 &&
            b.dtype() == torch::kFloat32 && y.dtype() == torch::kFloat32,
        "float32 only"
    );
    const int  N = x.size(0), In = x.size(1), Out = W.size(0);
    const int  threads = 256;
    const dim3 block(threads, 1, 1);
    const dim3 grid(div_up(Out, threads), N, 1);
    size_t     smem = In * sizeof(float);
    fc_forward_kernel<<<grid, block, smem>>>(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        N,
        In,
        Out
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fc kernel failed: ", cudaGetErrorString(err));
}

void linear(torch::Tensor x, torch::Tensor W, torch::Tensor b, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda() && b.is_cuda() && y.is_cuda(), "CUDA tensors req");
    TORCH_CHECK(
        x.dtype() == torch::kFloat32 && W.dtype() == torch::kFloat32 &&
            b.dtype() == torch::kFloat32 && y.dtype() == torch::kFloat32,
        "float32 only"
    );
    const int N = x.size(0), In = x.size(1), Out = W.size(0);
    // const int  threads = 256;
    // const dim3 block(threads, 1, 1);n hhmm,,,,,,,,,,,,,,,,,,,,,,,,,,,,
    // const dim3 grid(div_up(Out, threads), N, 1);

    dim3 blockDim(MATMUL_THREAD_M, MATMUL_THREAD_N);
    dim3 gridDim(CEIL_DIV(N, MATMUL_BLOCK_M), CEIL_DIV(Out, MATMUL_BLOCK_N));
    linear_forward<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        N,
        Out,
        In
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fc kernel failed: ", cudaGetErrorString(err));
}

void linear_fuse_if_forward(
    torch::Tensor x, torch::Tensor W, torch::Tensor b, torch::Tensor y, torch::Tensor v
) {
    TORCH_CHECK(x.is_cuda() && W.is_cuda() && b.is_cuda() && y.is_cuda(), "CUDA tensors req");
    TORCH_CHECK(
        x.dtype() == torch::kFloat32 && W.dtype() == torch::kFloat32 &&
            b.dtype() == torch::kFloat32 && y.dtype() == torch::kFloat32,
        "float32 only"
    );
    const int N = x.size(0), In = x.size(1), Out = W.size(0);
    // const int  threads = 256;
    // const dim3 block(threads, 1, 1);
    // const dim3 grid(div_up(Out, threads), N, 1);

    dim3 blockDim(MATMUL_THREAD_M, MATMUL_THREAD_N);
    dim3 gridDim(CEIL_DIV(N, MATMUL_BLOCK_M), CEIL_DIV(Out, MATMUL_BLOCK_N));
    linear_fuse_if<<<gridDim, blockDim>>>(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        y.data_ptr<float>(),
        v.data_ptr<float>(),
        N,
        Out,
        In
    );
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "fc kernel failed: ", cudaGetErrorString(err));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1_c1k5", &conv1_c1k5, "conv1 c1 k5");
    m.def("conv1_c1k5_implicit", &conv1_c1_k5_implicit, "conv1 c1 k5 implicit gemm");
    m.def("conv2_k5", &conv2_k5, "conv2 k5 general");
    m.def("pool2x2s2", &pool2x2s2, "maxpool2x2 stride2");
    m.def("if_integrate", &if_integrate, "IF integrate + fire");
    m.def("fc_forward", &fc_forward, "FC forward");
    m.def("linear", &linear, "Linear forward");
    m.def("linear_fuse_if_forward", &linear_fuse_if_forward, "Linear forward with if fusion");
}
