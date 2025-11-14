#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Tuple
import numpy as np


LAYERS: List[Tuple[str, int]] = [
    ("conv1", 8 * 24 * 24),
    ("if1", 8 * 24 * 24),
    ("pool1", 8 * 12 * 12),
    ("conv2", 16 * 8 * 8),
    ("if2", 16 * 8 * 8),
    ("pool2", 16 * 4 * 4),
    ("fc1", 128),
    ("if3", 128),
    ("fc2", 96),
    ("if4", 96),
    ("fc3", 10),
]


def load_bin(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Compare CUDA dumps against PyTorch reference.")
    ap.add_argument("root", nargs="?", default=".", help="Directory containing dbg_*.bin files")
    ap.add_argument("--tol", type=float, default=1e-5, help="Absolute tolerance for equality")
    ap.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for equality")
    ap.add_argument("--topk", type=int, default=5, help="Show top-K mismatched indices")
    args = ap.parse_args()

    root = args.root
    failures = []
    first_fail = None

    def p(cuda):
        return os.path.join(root, f"dbg_{cuda}.bin")

    def r(cuda):
        return os.path.join(root, f"dbg_{cuda}_ref.bin")

    print("Comparing layer dumps (CUDA vs PyTorch reference):")
    for name, expected in LAYERS:
        a_path, b_path = p(name), r(name)
        a = load_bin(a_path)
        b = load_bin(b_path)
        if a is None:
            print(f"- {name}: MISSING {a_path}")
            failures.append((name, "missing_cuda"))
            if first_fail is None:
                first_fail = name
            continue
        if b is None:
            print(f"- {name}: MISSING {b_path}")
            failures.append((name, "missing_ref"))
            if first_fail is None:
                first_fail = name
            continue
        if a.size != expected or b.size != expected:
            print(f"- {name}: SHAPE_MISMATCH cuda={a.size} ref={b.size} expected={expected}")
            failures.append((name, "shape"))
            if first_fail is None:
                first_fail = name
            continue

        diff = a - b
        abs_diff = np.abs(diff)
        max_abs = float(abs_diff.max()) if abs_diff.size > 0 else 0.0
        mean_abs = float(abs_diff.mean()) if abs_diff.size > 0 else 0.0
        denom = np.maximum(np.abs(b), 1e-12)
        max_rel = float(np.max(abs_diff / denom))
        ok = np.allclose(a, b, atol=args.tol, rtol=args.rtol)
        status = "PASS" if ok else "FAIL"
        print(f"- {name}: {status} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} max_rel={max_rel:.3e}")

        if not ok and args.topk > 0:
            # Show top-K mismatches
            idx = np.argsort(-abs_diff)[: args.topk]
            for j in idx:
                print(
                    f"  idx={int(j):6d} cuda={a[j]: .6e} ref={b[j]: .6e} abs={abs_diff[j]: .3e} rel={(abs_diff[j]/denom[j]): .3e}"
                )
            failures.append((name, "mismatch"))
            if first_fail is None:
                first_fail = name

    if first_fail is None:
        print("All layers match within tolerances.")
        sys.exit(0)
    else:
        print(f"First failing layer: {first_fail}")
        sys.exit(1)


if __name__ == "__main__":
    main()

