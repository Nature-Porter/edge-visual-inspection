# 使用手册：从 VS Code 运行 Edge Visual Inspection

本手册对应当前仓库的 Windows 研究原型。命令在项目根目录 `edge-visual-inspection` 的 VS Code PowerShell 终端执行。项目不负责下载依赖；已有本地资源和真实数据的位置见 [环境记录](ENVIRONMENT.md) 和 [数据说明](../data/README.md)。首次使用先按下文检查路径，再运行现成模型；日常回放不需要重新训练。

## 1. 先认识输入与产物

在线程序为 `build/inspection.exe`，通过 `scripts/run_windows.ps1` 启动。该脚本只为当前进程设置 OpenCV、CUDA、TensorRT 等 DLL 搜索路径，然后调用 C++ 程序；检测推理本身不启动 Python 子进程。

| 用途 | 本地文件 |
| --- | --- |
| FP32 主线 | `artifacts/metal_nut/fp32/model_manifest.json`、同目录 `model.onnx` |
| 经验证的 INT8/FP32 混合精度路线 | `artifacts/metal_nut/int8_selective_2_trt_compat/model_manifest.json`、`runs/trt-detailed-int8_selective_2_trt_compat/model.engine`、`reports/trt_verified_selective_2.json` |
| 固定回放清单 | `data/replay.json`；验证集 `data/validation.json` |
| 原始数据及清单 hash | `data/raw/metal_nut`、`reports/data_protocol.json` |
| 历史证据 | `reports/`、`runs/`、[验收表](ACCEPTANCE.md) |

manifest 绑定模型 SHA-256、输入输出契约、阈值和校准来源。TensorRT 还要求引擎与证据文件，并在运行前校验三者身份。请不要直接编辑这些文件后继续沿用旧的质量结论。

## 2. 检查环境并构建

机器环境记录的目标组合是 Windows、RTX 4060 8 GB、CUDA Toolkit 12.5、MSVC v143、OpenCV 4.11.0、C++ ONNX Runtime GPU 1.22.0、TensorRT 10.9.0.34。版本与绝对路径以**本机重新探测**为准，不要把历史配置当成当前事实。若切换电脑，先放好由你下载的依赖，再运行：

```powershell
Set-Location 'D:\Pythonproject\C++\edge-visual-inspection'
.\.venv-tools\Scripts\python.exe tools/environment.py --output reports/environment-current.json
.\scripts\probe_cuda.ps1
.\scripts\build_windows.ps1
```

`build_windows.ps1` 会通过 `vswhere` 找到 Visual Studio Build Tools、加载 x64 开发者环境、运行 CMake Release 构建和 CTest。普通 VS Code 终端里找不到 `cl.exe`，不等于没有安装编译器；先尝试这个脚本。现有 CMake 预设见 `CMakePresets.json`。如在其他机器安装了不同位置的 ORT/OpenCV，可用脚本参数 `-OrtRoot` 和 `-OpenCV_DIR` 指向本地 SDK；依赖详情见 [环境记录](ENVIRONMENT.md)。构建成功是代码可运行的证据，不代替 GPU 执行或质量验证。

## 3. 运行 FP32 GPU 检测

先选一个**新目录**，例如：

```powershell
$run = 'runs/manual-fp32-20261001-01'
.\scripts\run_windows.ps1 run `
  --manifest artifacts/metal_nut/fp32/model_manifest.json `
  --source data/replay.json `
  --backend ort-cuda `
  --postprocess cuda `
  --fps 20 `
  --deadline-ms 50 `
  --queue-capacity 4 `
  --output $run
$LASTEXITCODE
Get-Content "$run/summary.json" -Encoding UTF8
Get-Content "$run/events.jsonl" -Encoding UTF8 -TotalCount 3
```

PowerShell 的续行符是行尾反引号；复制时不要在它后面加空格。每次回放都换一个 `$run`：已有 `events.jsonl` 的目录会被拒绝，避免覆盖证据。`ort-cuda` 要求 CUDA Execution Provider 严格可用；`ort-cpu` 仅供 CPU 故障定位，不可当作 GPU 实测。`--postprocess cuda` 使用自研 CUDA 归约；`cpu` 使用数值参考。

`--source` 可接受 JSON 清单或图片目录。目录模式只枚举当前目录下小写扩展名 `.png`、`.jpg`、`.bmp` 的文件，不递归；按路径排序，逐图计算 SHA-256 作为 `source_image_id`。JSON 清单采用 `records` 数组，格式可直接参考 `data/replay.json`。如果导入自己的图片，需保证路径可读，并另建输出目录。自有图片可用于演示运行，但不能自动算作正式测试集。

要做固定时长循环回放，增加 `--loop --duration-seconds 60`；`--loop` 没有有限时长会被拒绝。正式历史 30 分钟回放已有证据，不建议只为查看程序而重复跑 1800 秒。

## 4. 运行 C++ TensorRT INT8 路线

这一路线使用经过层精度和运行 profile 审计的**直接 TensorRT 引擎**，不是只看 ONNX 里有没有 QDQ 节点。先确认上述三份文件存在，再执行：

```powershell
$run = 'runs/manual-trt-int8-20261001-01'
.\scripts\run_windows.ps1 run `
  --manifest artifacts/metal_nut/int8_selective_2_trt_compat/model_manifest.json `
  --source data/replay.json `
  --backend trt-int8 `
  --engine runs/trt-detailed-int8_selective_2_trt_compat/model.engine `
  --engine-evidence reports/trt_verified_selective_2.json `
  --postprocess cuda `
  --fps 20 `
  --deadline-ms 50 `
  --output $run
Get-Content "$run/summary.json" -Encoding UTF8
```

若要重做已验证的 69 张验证集审计，可把 `--source` 换成 `data/validation.json`，使用 `--fps 10 --deadline-ms 1000 --warmup 5`，再运行下文的事件校验。这样得到的是验证集证据，**不是**派生引擎新的独立最终测试。引擎通常和构建 GPU/软件栈绑定；更换设备或 TensorRT 版本时需重新构建并重新生成证据，不要复制旧结论。

## 5. 读懂运行结果

输出目录至少包含：

| 文件 | 怎么用 |
| --- | --- |
| `snapshot.json` | 保存本次参数、manifest、模型/引擎证据 hash、输入清单和调度计时方式，先用它确认“跑的是哪一个模型” |
| `events.jsonl` | 每帧一行 JSON；按 `frame_id` 检查覆盖、状态、决策与延迟 |
| `summary.json` | 计数、吞吐、P50/P95/P99、错误/溢出/超时、`latency_pass` |
| `map-*.tiff`、`heat-*.png` | 复检帧的原始异常图和可视化热图；图片写盘时间单列，不混入决策延迟 |

事件字段的权威约束是 [schema](../schemas/event.schema.json)。`decision` 是业务动作：`OK` 或 `REVIEW`；`status` 是运行状态：例如 `SUCCESS`、`TIMEOUT`、`QUEUE_OVERFLOW`、`INVALID_IMAGE`、`INFERENCE_ERROR`。成功并且分数**严格小于**图像阈值才是 `OK`；异常高分、未校准、超时、溢出、推理错误都进入 `REVIEW`。分数是原始 anomaly map 的最大值，**不是缺陷概率**。当前 `pixel_threshold=null`，热图可以观察定位，但项目没有像素级质量通过声明。

`scheduled_monotonic_ns` 是计划到帧时刻；`decision_latency_ms` 从此时刻到作出决策，包含排队、解码、预处理、推理和后处理。`model_gpu_ms` 是 CUDA 工作同步完成后的模型 GPU 计时，不包含整个决策链；`artifact_latency_ms` 是决策后生成文件的时间。过载时先看 `queue_overflow` 和事件 status，而不要仅看平均 FPS。

校验一份新回放：

```powershell
.\.venv-tools\Scripts\python.exe tools/validate_run.py --run $run --output reports/manual-run-validation.json
Get-Content reports/manual-run-validation.json -Encoding UTF8
```

`validate_run.py` 检查事件 schema、覆盖及终态；其输出路径也应按每次实验换新名字。`summary.json` 中的 `latency_pass` 只有在真实、已校准模型、帧覆盖完整、无溢出/错误且门槛满足时才可能为真。

## 6. 基准与性能解释

对一张已解码图片运行 100 次预热、1000 次计时：

```powershell
$bench = 'runs/manual-benchmark-20261001-01'
.\scripts\run_windows.ps1 benchmark `
  --manifest artifacts/metal_nut/fp32/model_manifest.json `
  --source data/replay.json `
  --backend ort-cuda `
  --postprocess cuda `
  --mode pipeline `
  --warmup 100 `
  --iterations 1000 `
  --output $bench
Get-Content "$bench/summary.json" -Encoding UTF8
```

`--mode model` 在计时前完成预处理；`pipeline` 每次计时包含 OpenCV 预处理、传输、推理和后处理，但**不含图片解码、调度、排队**。`benchmark --mode replay` 走有界队列调度路径，参数和输出解读与 `run` 相同。三种模式不能混成一种“端到端延迟”。FP32/INT8 同条件对照和边界已记录在 [基准协议](BENCHMARK_PROTOCOL.md) 及 [验收表](ACCEPTANCE.md)；不要把一次新机器的 benchmark 结果直接套用到历史报告。

ORT 后端可附加 `--profile` 并查看生成的 ORT profile，以验证实际 provider 分配；TensorRT 后端拒绝该参数，层级执行证据在引擎 evidence/profile 中。`requested_backend`、`declared_precision` 只描述请求，不能单独证明真实 GPU INT8 计算。

## 7. 离线训练、导出与质量复现

若只需要使用已训练模型，跳过本节。下列流程会消耗较长时间，且某些脚本会更新模型产物；请先阅读 [数据协议](../data/README.md)、[阶段状态](STATUS.md)、[研究报告](REPORT.md)，必要时在独立工作副本运行。所有下载仍由你手动完成。

```powershell
.\.venv-tools\Scripts\python.exe tools/prepare_data.py --config configs/metal_nut.yaml
.\.venv-train\Scripts\python.exe tools/train.py --config configs/metal_nut.yaml
.\.venv-train\Scripts\python.exe tools/export_onnx.py --config configs/metal_nut.yaml
.\.venv-train\Scripts\python.exe tools/verify_alignment.py --config configs/metal_nut.yaml --split validation --cpp
.\.venv-tools\Scripts\python.exe tools/evaluate.py --config configs/metal_nut.yaml --split validation
```

正式配置是 EfficientAD-S、10000 步；`--smoke` 只用于检查软件链路，不能参与正式质量结论。`data/train_normal.json` 用于训练，`stats_normal` 用于正常统计，`dev_normal` 与 `calibration` 用于开发，`validation` 用于阈值与候选选择。冻结记录在 `artifacts/frozen.json`；既有 `test` 结果已产生。**不要通过查看 `test` 结果再调阈值或筛层，然后仍把同一批图称为独立最终测试。** `tools/pipeline.py --stage P0` 到 `P8` 可审查对应阶段的命令、日志和产物；当前状态见 [STATUS](STATUS.md)。

## 8. 常见问题

| 现象 | 检查顺序 |
| --- | --- |
| VS Code 终端找不到 `cl.exe` | 先运行 `scripts/build_windows.ps1`；它会加载 x64 开发者环境。仍失败再核对 Build Tools/Windows SDK 安装与 `docs/ENVIRONMENT.md`。 |
| 缺 DLL、后端初始化失败 | 查 `reports/environment-current.json`、本地 SDK 目录与 `scripts/run_windows.ps1` 的当前进程 PATH。不要把 CPU 回退当成 GPU 验证。 |
| 退出码 2、模型/引擎 hash 不匹配 | 检查 manifest、ONNX、engine、evidence 是否来自同一次构建，以及输入输出形状和阈值契约。 |
| 输出目录已存在 | 换新 `--output`；旧结果作为证据保留。 |
| 大量 `TIMEOUT` 或 `QUEUE_OVERFLOW` | 看 `fps`、`deadline-ms`、`queue-capacity`，结合事件 `queue_ms`、`decode_ms` 和模型计时定位；不能靠调大 deadline 伪造原验收通过。 |
| 复检帧没有热图 | 查看事件 `artifact_error`；决策和写盘错误分别记录。 |

退出码：`0` 正常完成（即便存在业务上的 `REVIEW`）；`2` 配置/模型契约；`3` 后端或依赖；`4` 运行基础设施；`5` 验收失败。退出码 0 不表示质量、延迟、GPU 或 INT8 验收自动通过，需检查对应报告。
