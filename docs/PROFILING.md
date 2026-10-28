# perf 与 Nsight 性能分析

本页记录 2026-10-10 本机的实际探测和一次短采样。它是性能分析入口，不替代 [基准测试协议](BENCHMARK_PROTOCOL.md) 或既有质量验收。所有命令在项目根目录执行；没有自动下载依赖，也没有改动模型或冻结测试数据。

## 本机状态

| 工具 | 版本与路径 | 当前结果 |
|---|---|---|
| Nsight Systems | 2024.2.3，`C:/Program Files/NVIDIA Corporation/Nsight Systems 2024.2.3/target-windows-x64/nsys.exe` | 已采集正式 FP32 ONNX 的 C++ ORT CUDA `model` 短运行，报告 `reports/nsys-fp32-smoke.nsys-rep` |
| Nsight Compute | 2024.2.0，`C:/Program Files/NVIDIA Corporation/Nsight Compute 2024.2.0/target/windows-desktop-win7-x64/ncu.exe` | 已连接 `build/inspection.exe`，但 GPU 性能计数器权限被驱动拒绝；原始错误见 `reports/ncu-reduce-blocks-permission.log` |
| Linux perf | 本机没有可用 Linux 发行版：`wsl --list --verbose` 仅列出已停止的 `docker-desktop` | 尚未运行；没有 Linux `perf` 数据或 Linux 版程序构建证据 |

## Nsight Systems：已运行

```powershell
$nsysExe = 'C:/Program Files/NVIDIA Corporation/Nsight Systems 2024.2.3/target-windows-x64/nsys.exe'
& $nsysExe profile --trace=cuda --sample=none --output=reports/nsys-fp32-smoke powershell.exe -NoProfile -File scripts/run_windows.ps1 benchmark --manifest artifacts/metal_nut/fp32/model_manifest.json --source data/replay.json --backend ort-cuda --postprocess cuda --mode model --warmup 3 --iterations 10 --output runs/nsys-fp32-smoke
& $nsysExe stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum reports/nsys-fp32-smoke.nsys-rep
```

`reports/nsys-fp32-smoke-stats.txt` 保存了摘要。项目的 `inspection::reduce_blocks` 和 `inspection::reduce_final` 在 CUDA kernel 汇总中各出现 13 次（含 3 次预热）；中位持续时间分别约 3.84 µs 和 1.44 µs。报告也包含 cuDNN 卷积、内存拷贝等事件。`runs/nsys-fp32-smoke/summary.json` 显示 10 次测量、退出码 0、正式 FP32 模型 hash。Nsight Systems 提示 CPU context-switch trace 需要管理员权限，因此这次只声明 CUDA 时间线采集成功，不声明 CPU 上下文切换分析。分析报告可用 Nsight Systems 图形界面打开。

图形界面程序在 `C:/Program Files/NVIDIA Corporation/Nsight Systems 2024.2.3/host-windows-x64/nsys-ui.exe`；打开后用 **File → Open** 选择上面的 `.nsys-rep`。再次采样时，为 `--output` 和 `runs/` 指定新的后缀，保留本次原始报告。

这是短采样，采样器会影响执行；不能把其数值代替三轮、每轮 1000 次的正式基准，也不能将两个 kernel 时间相加后称为端到端收益。

## Nsight Compute：待 GPU 计数器授权

采集自研的第一级 CUDA 归约 kernel 的命令如下；运行时已定位到 `inspection.exe`，随后报 `ERR_NVGPUCTRPERM`，所以**没有生成有效的 kernel 指标报告**。

```powershell
$ncuExe = 'C:/Program Files/NVIDIA Corporation/Nsight Compute 2024.2.0/target/windows-desktop-win7-x64/ncu.exe'
& $ncuExe --target-processes all --kernel-name 'regex:reduce_blocks' --launch-count 1 --set basic --export reports/ncu-reduce-blocks-smoke powershell.exe -NoProfile -File scripts/run_windows.ps1 benchmark --manifest artifacts/metal_nut/fp32/model_manifest.json --source data/replay.json --backend ort-cuda --postprocess cuda --mode model --warmup 1 --iterations 3 --output runs/ncu-reduce-blocks-smoke
```

如需在当前普通用户会话中使用 Nsight Compute 的 GPU 性能计数器，由有管理员权限的人按 [NVIDIA 官方说明](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters)，在 **NVIDIA App → System → Advanced → Developer → Manage GPU Performance Counters** 中允许普通用户访问；该更改需要管理员权限。完成后重新执行上述命令。不要将权限报错或空报告记为已完成的 Nsight Compute 分析。

## Linux perf：需要 Linux 环境

`perf` 是 Linux 工具，不能直接分析当前 Windows `inspection.exe`。用户已要求所有下载由自己完成。若要开展 Linux CPU 分析，先自行按 [Microsoft WSL 官方安装说明](https://learn.microsoft.com/windows/wsl/install)安装一个 Linux 发行版（例如 Ubuntu）；随后还需在 Linux 中配置与该内核匹配的 `perf`、Linux 构建依赖及 Linux 版 ORT/CUDA，再构建和运行项目。当前仓库没有这些产物。获得可运行的 Linux `inspection` 后，用 `perf stat -d -- <检测命令>` 观察 CPU 计数器，用 `perf record -g -- <检测命令>` 与 `perf report` 定位 CPU 热点；具体可用事件和采样权限由 Linux 内核配置决定。参见 [Linux 内核 perf 指南](https://docs.kernel.org/admin-guide/workload-tracing.html)。

需要用户执行的最少安装命令是从 Windows PowerShell 运行 `wsl --install -d Ubuntu --location D:\WSL\Ubuntu`（建议目标目录；可按自己磁盘布局替换），完成首次启动和用户创建后再确认 `wsl --list --verbose`。若 WSL 无法提供所需的 `perf` 采样事件，应改用原生 Linux 主机；不能用 Windows 工具结果冒充 Linux `perf` 数据。
