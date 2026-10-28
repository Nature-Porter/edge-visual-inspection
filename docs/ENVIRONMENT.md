# Local environment record

See `reports/environment-current.json` for paths and versions captured by `tools/environment.py`. On 2026-09-30 the tested machine reported an RTX 4060 Laptop GPU, CUDA Toolkit 12.5.40, Visual Studio Build Tools v143 14.44.35207, Windows SDK at `D:/Windows Kits/10`, OpenCV 4.11.0 C++ SDK, and ORT GPU 1.22.0 C++ and Python SDKs. The driver advertises a newer CUDA runtime compatibility level; that number is not the Toolkit version.

`scripts/build_windows.ps1` imports the MSVC developer environment only into its process, selects v143 14.44, builds Release x64 with CMake/Ninja, and runs CTest. `scripts/run_windows.ps1` sets only the child process DLL search path for OpenCV and existing CUDA/cuDNN libraries. The training environment `.venv-train` reuses the existing `D:/anaconda/envs/py312` PyTorch CUDA 2.4.1 and torchvision 0.19.1 read-only; additional packages are in the project venv. The old `C:/Users/yuxiangw/ort-env` is untouched and supplies CUDA runtime DLLs for C++ use. CUDA 12.5 and PyTorch CUDA 12.4 are separate installed components.

User-managed downloads: Python dependencies in `requirements/train-reuse.txt`, ORT C++ SDK, OpenCV SDK, TensorRT SDK and official datasets/weights. `CMakeLists.txt` fails closed if the local nlohmann/json v3.11.3 header is absent. The project does not automatically download packages, data or weights.

Quantization uses `.venv-tools`, which already has `onnxconverter_common 1.16.0`. Python ORT CUDA in that environment needs the installed cuDNN sublibraries on its DLL search path. `tools/common.py` adds the existing Toolkit and `ort-env` NVIDIA DLL directories to that Python process only; smoke EfficientAD CUDA inference then completed with output shape `[1,1,256,256]`. No additional package was required.

Under WDDM, process-specific VRAM may not be available from `nvidia-smi`; a null field in summaries means not measured. Total GPU VRAM observation and process allocation are separate quantities. A provider name by itself is not execution evidence; ORT profiles and TensorRT engine layer records are required.
