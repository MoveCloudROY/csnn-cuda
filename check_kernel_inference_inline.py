#!/usr/bin/env python3
"""
Check a single CUDA kernel (compiled inline via PyTorch C++ extension) inside the
actual SCNN inference process with real FashionMNIST data and trained weights.

Supported ops to check: conv1, conv2, pool1, pool2, if1, if2, if3, if4, fc1, fc2, fc3

Examples:
  # Single sample
  python3 check_kernel_inference_inline.py . --op conv1 --img-idx 0 --timesteps 2

  # Batched check (e.g., 16 images starting at index 0)
  python3 check_kernel_inference_inline.py . --op conv1 --img-idx 0 --batch-size 16 --timesteps 2
"""

import argparse
import os
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


def load_weights(root):
    p = lambda name: os.path.join(root, name)
    weights = {
        'conv1_w': np.loadtxt(p('conv1.weight.txt'), dtype=np.float32),
        'conv1_b': np.loadtxt(p('conv1.bias.txt'), dtype=np.float32),
        'conv2_w': np.loadtxt(p('conv2.weight.txt'), dtype=np.float32),
        'conv2_b': np.loadtxt(p('conv2.bias.txt'), dtype=np.float32),
        'fc1_w':   np.loadtxt(p('fc1.weight.txt'), dtype=np.float32),
        'fc1_b':   np.loadtxt(p('fc1.bias.txt'), dtype=np.float32),
        'fc2_w':   np.loadtxt(p('fc2.weight.txt'), dtype=np.float32),
        'fc2_b':   np.loadtxt(p('fc2.bias.txt'), dtype=np.float32),
        'fc3_w':   np.loadtxt(p('fc3.weight.txt'), dtype=np.float32),
        'fc3_b':   np.loadtxt(p('fc3.bias.txt'), dtype=np.float32),
    }
    weights['conv1_w'] = weights['conv1_w'].reshape(8, 1, 5, 5)
    weights['conv2_w'] = weights['conv2_w'].reshape(16, 8, 5, 5)
    weights['fc1_w'] = weights['fc1_w'].reshape(128, 256)
    weights['fc2_w'] = weights['fc2_w'].reshape(96, 128)
    weights['fc3_w'] = weights['fc3_w'].reshape(10, 96)
    return {k: torch.from_numpy(v) for k, v in weights.items()}


def read_images(root, start_idx=0, batch_size=1):
    img_path = os.path.join('data', 'FashionMNIST', 'raw', 't10k-images-idx3-ubyte')
    lab_path = os.path.join('data', 'FashionMNIST', 'raw', 't10k-labels-idx1-ubyte')
    if batch_size <= 0:
        raise ValueError('batch_size must be >= 1')
    # Images
    with open(img_path, 'rb') as f:
        f.seek(16 + start_idx * 28 * 28)
        buf = f.read(batch_size * 28 * 28)
    imgs = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    imgs = (imgs / 255.0 - 0.5) / 0.5
    imgs = torch.from_numpy(imgs).view(batch_size, 1, 28, 28)
    # Labels
    with open(lab_path, 'rb') as f:
        f.seek(8 + start_idx)
        lab_buf = f.read(batch_size)
    labels = torch.tensor(list(lab_buf), dtype=torch.long)
    return imgs, labels


def build_module():
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernel')
    wrapper = os.path.join(dir, 'torch_wrapper.cu')
    src = os.path.join(dir, 'inline_infer_kernels.cu')
    linear_src = os.path.join(dir, 'linear.cu')
    m = load(
        name="inline_infer_kernels",
        sources=[wrapper, src, linear_src],
        verbose=True,
        with_cuda=True,
        extra_include_paths=[dir],
        extra_cuda_cflags=["-O3", "-std=c++17"], # , "-gencode=arch=compute_70,code=sm_70"
        extra_cflags=["-O3", "-std=c++17"],
    )
    return m


def compare(a: torch.Tensor, b: torch.Tensor, name: str, atol=1e-5, rtol=1e-4):
    diff = (a - b).abs()
    max_abs = diff.max().item() if diff.numel() > 0 else 0.0
    sum = diff.sum().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    print(f"{name}: max_abs={max_abs:.6g}, sum={sum} -> {'OK' if ok else 'MISMATCH'}")
    return ok

def save(a: torch.Tensor, b: torch.Tensor, name: str):
    # save a and b to files
    import matplotlib.pyplot as plt
    
    a_np = a.cpu().numpy()
    b_np = b.cpu().numpy()
    
    # Save as text files (保持shape)
    np.savetxt(f"{name}_a.txt", a_np.reshape(a_np.shape[0], -1))
    np.savetxt(f"{name}_b.txt", b_np.reshape(b_np.shape[0], -1))
    
    # Save as heatmap images (热力图) - only last 2 dimensions (H, W)
    # Take a single slice instead of averaging
    # if len(a_np.shape) >= 2:
    #     # Extract a single (H, W) slice - use index [0, 0, ...] for all dimensions except last 2
    #     idx = tuple([0] * (len(a_np.shape) - 2))
    #     a_slice = a_np[idx]  # Shape: (H, W)
    #     b_slice = b_np[idx]  # Shape: (H, W)
        
    #     # fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    #     im0 = plt.imshow(a_slice, cmap='viridis', aspect='auto')
    #     plt.title(f'{name}_a {idx}+(H,W)')
    #     plt.colorbar(im0)
        
    #     # im1 = axes[1].imshow(b_slice, cmap='viridis', aspect='auto')
    #     # axes[1].set_title(f'{name}_b {idx}+(H,W)')
    #     # plt.colorbar(im1, ax=axes[1])
        
    #     # im2 = axes[2].imshow(np.abs(a_slice - b_slice), cmap='hot', aspect='auto')
    #     # axes[2].set_title(f'{name}_diff {idx}+(H,W)')
    #     # plt.colorbar(im2, ax=axes[2])
        
    #     plt.tight_layout()
    #     plt.savefig(f"{name}_heatmap.png", dpi=300, bbox_inches='tight')
    #     plt.close()
    print(f"{name}: save data to {name}_a.txt, {name}_b.txt and {name}_heatmap.png")


def run_check(root, op, img_idx=0, T=2, batch_size=1, save_op=None):
    save_op = save_op or ['none']
    if not torch.cuda.is_available():
        print("CUDA unavailable.")
        return 1
    mod = build_module()

    # Load weights and data
    W = load_weights(root)
    x, labels = read_images(root, img_idx, batch_size)

    # padding
    # out = torch.zeros(1008, 1, 28, 28)
    # out[:1000] = x
    out = x
    
    x = out.cuda().float()

    


    # Convert weights to CUDA
    W = {k: v.cuda().float() for k, v in W.items()}

    # Membranes
    N = x.size(0)
    v1_ref = torch.zeros((N, 8, 24, 24), device='cuda', dtype=torch.float32)
    v2_ref = torch.zeros((N, 16, 8, 8), device='cuda', dtype=torch.float32)
    v3_ref = torch.zeros((N, 128), device='cuda', dtype=torch.float32)
    v4_ref = torch.zeros((N, 96), device='cuda', dtype=torch.float32)
    v1 = torch.zeros((N, 8, 24, 24), device='cuda', dtype=torch.float32)
    v2 = torch.zeros((N, 16, 8, 8), device='cuda', dtype=torch.float32)
    v3 = torch.zeros((N, 128), device='cuda', dtype=torch.float32)
    v4 = torch.zeros((N, 96), device='cuda', dtype=torch.float32)


    logits_acc_ref = torch.zeros((N, 10), device='cuda', dtype=torch.float32)
    logits_acc_tst = torch.zeros_like(logits_acc_ref)

    ok_all = True

    for t in range(T):
        # conv1
        y_ref = F.conv2d(x, W['conv1_w'], W['conv1_b'], stride=1, padding=0)
        y_tst = torch.empty_like(y_ref)
        if 'conv1' in op or 'all' in op:
            mod.conv1_c1k5(x, W['conv1_w'], W['conv1_b'], y_tst)
            ok_all &= compare(y_tst, y_ref, f"t={t} conv1")
        else:
            y_tst.copy_(y_ref)

        # IF1
        s1_ref = (v1_ref + y_ref >= 1.0).float(); v1_ref = torch.where(s1_ref > 0, torch.zeros_like(v1_ref), v1_ref + y_ref)
        s1_tst = torch.empty_like(s1_ref)
        if 'if1' in op or 'all' in op:
            mod.if_integrate(y_tst, v1, s1_tst, 1.0)
            ok_all &= compare(s1_tst, s1_ref, f"t={t} if1")
        else:
            s1_tst.copy_(s1_ref)

        # pool1
        p1_ref = F.max_pool2d(s1_ref, 2, 2)
        p1_tst = torch.empty_like(p1_ref)
        if 'pool1' in op or 'all' in op:
            mod.pool2x2s2(s1_tst, p1_tst)
            ok_all &= compare(p1_tst, p1_ref, f"t={t} pool1")
        else:
            p1_tst.copy_(p1_ref)

        if 'pool1' in save_op or 'all' in save_op:
            save(p1_ref, p1_tst, f"t{t}_pool1")

        # conv2
        y2_ref = F.conv2d(p1_ref, W['conv2_w'], W['conv2_b'], stride=1, padding=0)
        y2_tst = torch.empty_like(y2_ref)
        if 'conv2' in op or 'all' in op:
            mod.conv2_k5(p1_tst, W['conv2_w'], W['conv2_b'], y2_tst)
            ok_all &= compare(y2_tst, y2_ref, f"t={t} conv2")
        else:
            y2_tst.copy_(y2_ref)

        # IF2
        s2_ref = (v2_ref + y2_ref >= 1.0).float(); v2_ref = torch.where(s2_ref > 0, torch.zeros_like(v2_ref), v2_ref + y2_ref)
        s2_tst = torch.empty_like(s2_ref)
        if 'if2' in op or 'all' in op:
            mod.if_integrate(y2_tst, v2, s2_tst, 1.0)
            ok_all &= compare(s2_tst, s2_ref, f"t={t} if2")
        else:
            s2_tst.copy_(s2_ref)
        
        if 'if2' in save_op or 'all' in save_op:
            save(s2_ref, s2_tst, f"t{t}_if2")

        # pool2
        p2_ref = F.max_pool2d(s2_ref, 2, 2)
        p2_tst = torch.empty_like(p2_ref)
        if 'pool2' in op or 'all' in op:
            mod.pool2x2s2(s2_tst, p2_tst)
            ok_all &= compare(p2_tst, p2_ref, f"t={t} pool2")
        else:
            p2_tst.copy_(p2_ref)

        if 'pool2' in save_op or 'all' in save_op:
            save(p2_ref, p2_tst, f"t{t}_pool2")

        # flatten
        f_ref = p2_ref.view(N, -1)
        f_tst = p2_tst.view(N, -1)

        # fc1
        # start_torch = torch.cuda.Event(enable_timing=True)
        # end_torch = torch.cuda.Event(enable_timing=True)
        # start_tst = torch.cuda.Event(enable_timing=True)
        # end_tst = torch.cuda.Event(enable_timing=True)

        # start_torch.record()
        fc1_ref = F.linear(f_ref, W['fc1_w'], W['fc1_b'])
        # end_torch.record()
        # torch.cuda.synchronize()
        # print(f"  fc1 (PyTorch) time: {start_torch.elapsed_time(end_torch):.3f} ms")
        fc1_tst = torch.empty_like(fc1_ref)
        if 'fc1' in op or 'all' in op:
            # start_tst.record()
            if 'fuse3' in op:
                mod.linear_fuse_if_forward(f_tst, W['fc1_w'], W['fc1_b'], fc1_tst, v3)
            else:
                mod.linear(f_tst, W['fc1_w'], W['fc1_b'], fc1_tst)
                ok_all &= compare(fc1_tst, fc1_ref, f"t={t} fc1")
            # end_tst.record()
            # torch.cuda.synchronize()
            # print(f"  fc1 (Custom)  time: {start_tst.elapsed_time(end_tst):.3f} ms")
            
        else:
            fc1_tst = F.linear(f_tst, W['fc1_w'], W['fc1_b'])
        
        if 'fc1' in save_op or 'all' in save_op:
            save(fc1_ref, fc1_tst, f"t{t}_fc1")

        # IF3
        s3_ref = (v3_ref + fc1_ref >= 1.0).float(); v3_ref = torch.where(s3_ref > 0, torch.zeros_like(v3_ref), v3_ref + fc1_ref)
        s3_tst = torch.empty_like(s3_ref)
        if 'if3' in op or 'all' in op:
            if 'fuse3' in op:
                s3_tst = fc1_tst
            else:
                mod.if_integrate(fc1_tst, v3, s3_tst, 1.0)
            ok_all &= compare(s3_tst, s3_ref, f"t={t} if3")
        else:
            s3_tst = (v3 + fc1_tst >= 1.0).float(); v3 = torch.where(s3_tst > 0, torch.zeros_like(v3), v3 + fc1_tst)

        if 'if3' in save_op or 'all' in save_op:
            save(s3_ref, s3_tst, f"t{t}_if3")

        # fc2
        fc2_ref = F.linear(s3_ref, W['fc2_w'], W['fc2_b'])
        fc2_tst = torch.empty_like(fc2_ref)
        if 'fc2' in op or 'all' in op:
            if 'fuse4' in op:
                # Use fused version for fc2 + if4
                mod.linear_fuse_if_forward(s3_tst, W['fc2_w'], W['fc2_b'], fc2_tst, v4)
            else:   
                mod.linear(s3_tst, W['fc2_w'], W['fc2_b'], fc2_tst)
                ok_all &= compare(fc2_tst, fc2_ref, f"t={t} fc2")
        else:
            fc2_tst = F.linear(s3_tst, W['fc2_w'], W['fc2_b'])

        # IF4
        s4_ref = (v4_ref + fc2_ref >= 1.0).float(); v4_ref = torch.where(s4_ref > 0, torch.zeros_like(v4_ref), v4_ref + fc2_ref)
        s4_tst = torch.empty_like(s4_ref)
        if 'if4' in op or 'all' in op:
            if 'fuse4' in op:
                s4_tst = fc2_tst
            else:
                mod.if_integrate(fc2_tst, v4, s4_tst, 1.0)
            ok_all &= compare(s4_tst, s4_ref, f"t={t} if4")
        else:
            s4_tst = (v4 + fc2_tst >= 1.0).float(); v4 = torch.where(s4_tst > 0, torch.zeros_like(v4), v4 + fc2_tst)

        if 'if4' in save_op or 'all' in save_op:
            save(s4_ref, s4_tst, f"t{t}_if4")

        # fc3 (logits increment)
        fc3_ref = F.linear(s4_ref, W['fc3_w'], W['fc3_b'])
        fc3_tst = torch.empty_like(fc3_ref)
        if 'fc3' in op or 'all' in op:
            mod.linear(s4_tst, W['fc3_w'], W['fc3_b'], fc3_tst)     # or .abs().sum()
            print(f"###############, {fc3_tst.data_ptr() % 16}")
            ok_all &= compare(fc3_tst, fc3_ref, f"t={t} fc3")
        else:
            fc3_tst = F.linear(s4_tst, W['fc3_w'], W['fc3_b'])

        logits_acc_ref += fc3_ref
        logits_acc_tst += fc3_tst
        # print(logits_acc_tst)

    # logits_acc_ref = logits_acc_ref[:1000]
    # logits_acc_tst = logits_acc_tst[:1000]
    # Final average and prediction
    logits_ref = logits_acc_ref / float(T)
    logits_tst = logits_acc_tst / float(T)
    ok_all &= compare(logits_tst, logits_ref, "final logits (avg)")
    pred_ref = torch.argmax(logits_ref, dim=1)
    pred_tst = torch.argmax(logits_tst, dim=1)
    labels = labels.cuda()
    acc_ref = (pred_ref == labels).float().mean().item()
    acc_tst = (pred_tst == labels).float().mean().item()
    agree = (pred_ref == pred_tst).all().item() if pred_ref.numel() > 0 else True
    # Print compact summary
    print(f"batch={N}, idx=[{img_idx}..{img_idx+N-1}], acc_ref={acc_ref:.4f}, acc_tst={acc_tst:.4f}, preds_agree={bool(agree)}")
    if not agree:
        ok_all = False
    print("Result:", "OK" if ok_all else "MISMATCH")
    return 0 if ok_all else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='model_90_04_fp32', help='root directory containing weights and data (e.g., repo root)')
    # parser.add_argument('--op', default='none', choices=['none', 'all', 'conv1','conv2','pool1','pool2','if1','if2','if3','if4','fc1','fc2','fc3'])
    parser.add_argument('--op', nargs='*', default=['none'], choices=['none', 'all', 'conv1','conv2','pool1','pool2','if1','if2','if3','if4','fc1','fc2','fc3', 'fuse1', 'fuse2', 'fuse3','fuse4', 'fuse5'])
    parser.add_argument('--img-idx', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--timesteps', type=int, default=2)
    # parser.add_argument('--heat', action='store_true', help='save heatmap images for comparison')
    parser.add_argument('--save', nargs='*', default=['none'], choices=['none', 'all', 'conv1','conv2','pool1','pool2','if1','if2','if3','if4','fc1','fc2','fc3', 'fuse1', 'fuse2', 'fuse3','fuse4', 'fuse5'])

    parser.add_argument('--profiler', action='store_true', help='enable profiler')

    args = parser.parse_args()
    sys.exit(run_check(args.root, args.op, img_idx=args.img_idx, T=args.timesteps, batch_size=args.batch_size, save_op=args.save))


if __name__ == '__main__':
    main()
