#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F


def load_weights(root):
    # Match paths used by infer.cu main
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
    # Reshape tensors
    weights['conv1_w'] = weights['conv1_w'].reshape(8, 1, 5, 5)
    weights['conv2_w'] = weights['conv2_w'].reshape(16, 8, 5, 5)
    weights['fc1_w'] = weights['fc1_w'].reshape(128, 256)
    weights['fc2_w'] = weights['fc2_w'].reshape(96, 128)
    weights['fc3_w'] = weights['fc3_w'].reshape(10, 96)
    return {k: torch.from_numpy(v) for k, v in weights.items()}


def read_first_image(root):
    # Read raw FashionMNIST test image 0 from ubyte
    img_path = os.path.join(root, 'data', 'FashionMNIST', 'raw', 't10k-images-idx3-ubyte')
    with open(img_path, 'rb') as f:
        f.read(16)  # header
        buf = f.read(28*28)
    img = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    img = (img / 255.0 - 0.5) / 0.5
    img = torch.from_numpy(img).view(1, 1, 28, 28)
    return img


def dump(name, t):
    np.asarray(t.detach().cpu().reshape(-1), dtype=np.float32).tofile(name)


def main():
    if len(sys.argv) < 2:
        print('Usage: python verify_with_pytorch.py <root_dir>')
        sys.exit(1)
    root = sys.argv[1]
    w = load_weights(root)
    x = read_first_image(root)

    # Timestep t=0 only for debugging
    # IFNode hard reset (v_reset=0), v_th=1.0
    v1 = torch.zeros((1, 8, 24, 24))
    v2 = torch.zeros((1, 16, 8, 8))
    v3 = torch.zeros((1, 128))
    v4 = torch.zeros((1, 96))

    # conv1
    y = F.conv2d(x, w['conv1_w'], w['conv1_b'], stride=1, padding=0)
    dump('dbg_conv1_ref.bin', y)
    # IF1 hard reset
    v1 = v1 + y
    s1 = (v1 >= 1.0).float()
    v1 = torch.where(s1 > 0, torch.zeros_like(v1), v1)
    dump('dbg_if1_ref.bin', s1)
    # pool1
    y = F.max_pool2d(s1, kernel_size=2, stride=2)
    dump('dbg_pool1_ref.bin', y)

    # conv2
    y = F.conv2d(y, w['conv2_w'], w['conv2_b'], stride=1, padding=0)
    dump('dbg_conv2_ref.bin', y)
    # IF2
    v2 = v2 + y
    s2 = (v2 >= 1.0).float()
    v2 = torch.where(s2 > 0, torch.zeros_like(v2), v2)
    dump('dbg_if2_ref.bin', s2)
    # pool2
    y = F.max_pool2d(s2, kernel_size=2, stride=2)
    dump('dbg_pool2_ref.bin', y)

    # flatten
    y = y.view(1, -1)
    # fc1
    y = F.linear(y, w['fc1_w'], w['fc1_b'])
    dump('dbg_fc1_ref.bin', y)
    # IF3
    v3 = v3 + y
    s3 = (v3 >= 1.0).float()
    v3 = torch.where(s3 > 0, torch.zeros_like(v3), v3)
    dump('dbg_if3_ref.bin', s3)

    # fc2
    y = F.linear(s3, w['fc2_w'], w['fc2_b'])
    dump('dbg_fc2_ref.bin', y)
    # IF4
    v4 = v4 + y
    s4 = (v4 >= 1.0).float()
    v4 = torch.where(s4 > 0, torch.zeros_like(v4), v4)
    dump('dbg_if4_ref.bin', s4)

    # fc3
    y = F.linear(s4, w['fc3_w'], w['fc3_b'])
    dump('dbg_fc3_ref.bin', y)

    print('Dumped PyTorch reference layer outputs for t=0.')


if __name__ == '__main__':
    main()
