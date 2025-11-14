# Repository Guidelines

## Project Structure & Module Organization
- `infer.cu`: CUDA/C++ inference entrypoint. Loads FashionMNIST test data from `<dir>/data/FashionMNIST/raw` and model parameters (`conv*.txt`, `fc*.txt`) from `<dir>`. Run with a path argument to that directory.
- `train_9004.py`: PyTorch + SpikingJelly training script. Downloads FashionMNIST into `data/` and exports learned parameters as flattened `.txt` files in the script directory.
- `model_90_04_fp32/`: Example exported parameters/checkpoint folder (optional reference).

## Build, Test, and Development Commands
- Build inference (requires CUDA Toolkit + cuDNN):
  - `nvcc -O3 -std=c++14 -lcudnn infer.cu -o infer` (add `-arch=sm_75` or your GPU’s SM).
- Run inference (from repo root after data/params exist):
  - `./infer .` → prints `seconds:accuracy`.
- Train (optional):
  - `python3 train_9004.py` (needs `torch`, `torchvision`, `spikingjelly`). For a smoke run, temporarily set `EPOCHS = 1` and reduce `BATCH_SIZE` in the file.

## Coding Style & Naming Conventions
- CUDA/C++: C++14, 4-space indent. Use `snake_case` for functions/vars, `MACRO_CASE` for constants/macros. Do not edit regions marked “DO NOT MODIFY”. Keep GPU memory handling explicit and guarded with `checkCudaErrors`.
- Python: Follow PEP 8, 4-space indent. Classes `PascalCase`, functions/variables `snake_case`. Preserve seeding and determinism patterns in training.
- Formatting tools are not enforced; if used, prefer `clang-format` (CUDA) and `black`/`ruff` (Python).

## Testing Guidelines
- No formal test suite yet. Add lightweight checks:
  - Inference smoke test: `./infer .` and verify non-trivial accuracy (≥80% with well-trained weights).
  - Determinism: ensure seeds remain set; log `seconds:accuracy` and GPU info.
- When adding kernels/layers, include a minimal reproduction in the PR description or a short script.

## Commit & Pull Request Guidelines
- Commits: Use Conventional Commits (`feat:`, `fix:`, `perf:`, `docs:`, `refactor:`). Keep changes focused; explain rationale and any perf impact in the body.
- PRs: Provide summary, linked issues, build/run commands, and sample output (`seconds:accuracy`). Note environment (GPU model, driver, CUDA/cuDNN versions). Include before/after benchmarks for performance changes.

## Security & Configuration Tips
- Verify environment: `nvcc --version`, `nvidia-smi`. Adjust NVCC arch (e.g., `-arch=sm_75`) for your GPU.
- Keep datasets and large exported weight files out of version control.
