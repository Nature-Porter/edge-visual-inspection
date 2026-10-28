# 决策记录

- 2026-10-03：按用户新的界面需求增加**可选** Qt Widgets 5.15 C++ 前端，复用本机 `D:/anaconda/Library` 的 MSVC SDK，不下载、不复制 Qt。界面通过 `QProcess` 直接启动已有 `inspection.exe`，仅读取 `events.jsonl`、`summary.json` 和复检热图；在线推理、决策及证据链继续由原 C++ 程序负责。离线训练/导出命令在界面中仅供阅读与复制，避免一键重跑覆盖冻结产物；中止回放会标记为不可验收。原无界面构建保持默认。用户随后明确取消 VisA 数据切换，因此保留 `metal_nut` 数据协议，不写入假 VisA 质量指标。验证见 `reports/gui-build.log`、`reports/gui-preview.png`、`reports/gui-smoke-validation.json`。

- 2026-09-30：在已通过验证集质量和层精度审计的 `int8_selective_2_trt_compat` 引擎上，增加 C++17 直接 TensorRT 运行后端 `trt-int8`；运行时必须指定引擎和审计文件，启动前比对模型/引擎/证据 hash，TensorRT I/O 名称、dtype、shape 与清单一致才进入回放。CUDA stream 上记录 enqueueV3 前后事件并同步，保持原有决策时钟、有界队列、错误复检和 CUDA 后处理。此处复用本地 TensorRT 10.9 SDK，无下载；69 张 validation 的 C++ 分数与 `trtexec` 最大差 3.3379e-6，CPU/CUDA 后处理完全一致，错误 FP32 引擎被拒。证据 `reports/trt-cpp-validation-quality.json`、`reports/trt-cpp-wrong-engine.log`。原 `reports/trt_verified_selective_2.json` 继续作为不可回写的引擎审计输入，其历史 `cpp_trt_integration_verified=false` 指审计生成时状态；新增 C++ 证据单独记录。派生引擎的独立最终 test 仍未运行，不沿用源 ONNX 指标。
- 2026-09-30：C++ INT8 的 30 分钟回放 36001/36001 帧、决策 P95 32.5113 ms、0 错误/溢出、5 超时，见 `reports/trt-cpp-soak-audit.json`。本次未在回放期间采样进程工作集，因此仅声明事件和时间门槛通过，不把 FP32 的资源稳定性结果转移给 INT8；WDDM 下仍无可靠进程级 VRAM。重复帧仅证明运行稳定性，不新增质量样本。
- 2026-09-30：使用已有 FP32 TensorRT 引擎增加 `trt-fp32` 比较入口，严格要求其模型/引擎/构建证据 hash 和 FP32 精度意图一致。三轮 C++ pipeline 对照得到 2.619× wall P50 比值，但该模式不含调度/队列/解码；FP32 TensorRT 验证集沿用 ORT 阈值时 FNR 0.03448，超过相对增量 0.02，只保留为性能对照。报告 `reports/trt-cpp-benchmark-audit.json` 明确 `both_routes_quality_pass=false`、`decision_latency_speedup_claim=false`；不使用最终 test 为新引擎选阈值。
- 2026-09-30：另做 300 秒 INT8 资源补测并同步采样 27 次；进程工作集范围 295.68–298.17 MiB、首末 +1.31 MiB，设备全局显存首末 +1 MiB，见 `reports/trt-cpp-memory-300s-summary.json`。样本只覆盖 271.203 秒，不外推为 30 分钟资源结果；未测到进程级 VRAM。TensorRT DLL 已加入 `tools/integration_tests.py` 的进程 PATH，重新运行 10/10 通过。

- 2026-09-30 用户明确要求：所有下载交给用户，代理不得自行下载。已停止训练环境 pip 下载；CMake 移除 FetchContent 自动获取，资源脚本仅校验/解压本地文件，训练启动先检查本地辅助资源。后续缺件列官方来源、版本、目标路径；仅可使用已有文件或通过 --no-index 从用户提供的本地 wheel 安装。

- 2026-09-30：本机 v143 14.44.35207 + nvcc 12.5.40 已编译并运行 CUDA add_one=42（reports/cuda-smoke.log），未加入 unsupported-compiler 开关。VS 18 安装器默认选 14.51，构建脚本显式选择 14.44。Windows SDK 位于 D:/Windows Kits/10。
- 训练下载 2.7.1 wheel 失败后发现已有 D:/anaconda/envs/py312 中 torch 2.4.1+cu124 / torchvision 0.19.1+cu124 且 CUDA 可用。新增 .venv-train（system-site-packages）复用其只读安装，所有新增包局限到 .venv-train；先验证 Anomalib 2.1.0 依赖约束与真实 smoke。原环境不升级、不卸载。
- ORT 用户解压包存在一层同名嵌套，构建脚本自动定位，不移动用户 SDK。JSON 采用官方 v3.11.3 单头文件；下载完整性与 hash 保存到报告。

- 依赖解析发现 onnxconverter-common 1.14.0 固定 protobuf 3.20.2，与所选 ONNX 冲突；改为 1.16.0（安装日志保留）。项目 ORT 固定 1.22.0，为官方明确 TRT 10.9 组合，C++ SDK 使用同版本；这不代表 TensorRT 已验证。

- 2026-09-29：使用项目 .venv（已有 Python 3.11.10），不修改 C:/Users/yuxiangw/ort-env。历史 ORT 1.26.0 已读取确认；驱动报告 CUDA 13.3 不代表 nvcc 版本，实际 nvcc 为 12.5.40。
- 训练固定 Anomalib 2.1.0，优先官方 EfficientAd Lightning 实现，不重新实现损失/网络。外部 OpenCV INTER_LINEAR / RGB / float32 / NCHW / [0,1]；训练与部署统一；ImageNet Normalize 留在 PDN 内。此预处理是项目候选，不能宣称默认 torchvision antialias 基线等价。
- 精度对齐在测试前固定 atol=1e-5、rtol=1e-4，关闭 TF32。未校准阈值禁止 OK；像素阈值可 null，不编造像素质量。
- ORT 官方 TRT 矩阵目前明确列出 ORT 1.22 + TensorRT 10.9 + CUDA 12.0–12.8；不能推断 1.26 支持。若试验采用该组合，隔离环境，保留 1.26。
- 最终测试仅 P7 冻结后执行；回放仅 validation。阈值候选中先最大召回，再最小 FPR，最后取较高阈值，固定 deterministic tie-break。
- 正式 FP32 验证集 69 张得到 AUROC 0.97931、FNR 0、FPR 0.05；量化筛选只用 validation，MinMax 与 selective_1 未过预设门槛，percentile 与 selective_2/3 通过。P4 是候选图与品质验证，不代表执行精度。正式 C++ CUDA profile 的 3000 个节点事件全部在 CUDAExecutionProvider；P6 三轮及交替 A/B 的后处理收益与整管线变动分别记录在 reports/postprocess_ab.json。
- ORT TRT 的原始 percentile QDQ 已由 TensorRT Provider 执行主子图，CPU 回退有 22 个小张量 DequantizeLinear、无 CPU Conv/MatMul；但 ORT 缓存引擎使用层名级别的 profiling verbosity，`trtexec --loadEngine --exportLayerInfo` 只有 61 个层名，无精度元数据。根据 NVIDIA Engine Inspector 文档，加载时加 `--profilingVerbosity=detailed` 不能补回构建时未保存的字段。因此该缓存引擎不能单独证明 INT8，不凭 QDQ 名称、provider 或量化权重名宣称通过；后续独立详细引擎证据单列。
- 首次 1800 秒回放 36000/36000 帧完成、0 错误/溢出、3 次超时，但决策 P95 40.3861 ms，旧 `std::this_thread::sleep_until` 的 queue_ms P95 约 15 ms；实际结束时长 1799.983 秒。Windows 11 上改用进程内高精度 waitable timer（失败时回退标准 sleep），并把终点帧纳入回放以保证时长至少 1800 秒。此更改不涉及模型、预处理、阈值或冻结测试品质；69 帧短测 P95 22.421 ms。完整复测 36001/36001 帧、1800.026 秒、P95 26.7411 ms、1 次超时，见 `runs/soak-highres/summary.json` 与 `reports/soak-highres-validation.json`。依据 Microsoft CreateWaitableTimerExW 文档。
- C++ ORT CUDA 后端继续禁止静默 CPU EP 回退。FP32 与 FP16 可初始化并运行；三个 QDQ 候选由于存在 CPU 分配节点均初始化失败，详见 `reports/precision_benchmarks.json` 与各失败日志。独立 Python ORT TensorRT 探针允许已审计的 22 个小张量 DequantizeLinear CPU 回退，但既无层精度证据，也未在 C++ 主程序提供相同受控回退，故不把 QDQ 候选标为已部署。FP32/FP16 三轮同后端基准的模型 GPU P50 中位数为 12.6258/12.7582 ms，FP16 未带来可验证的延迟收益。
- 直接 `trtexec` 对冻结 percentile QDQ ONNX 因静态 bias DQ 解析失败。在独立副本上只折叠 22 个静态 bias DQ；69 张 validation 与源图输出完全相同。该副本 TensorRT 详细 engine 有 21 个 INT8 卷积的 datatype、权重与 `i8i8` tactic，并与运行 profile 对应，但直接 TensorRT 与 ORT 在首张验证图的输出最大差 0.17016；该 percentile 引擎未作质量通过声明，也不回写冻结模型。命令、hash 和原始证据见 `tools/fold_qdq_bias.py`、`tools/check_graph_equivalence.py`、`tools/build_trt_detailed.py`、`tools/run_trt_detailed.py`、`tools/verify_trt_detailed_runtime.py`、`reports/trt_detailed_exploratory.json`。
- 三个冻结 QDQ 候选各自的缓存 TensorRT Provider 引擎运行 69 张 validation，固定原阈值，不用最终 test。`percentile` FNR 0.03448、`selective_3` FNR 0.06897，均超过相对 FP32 最大增量 0.02；`selective_2` FNR 0、FPR 0.05、AUROC 0.97672，验证质量通过但层精度未验证。逐图分数及 profile 见 `reports/trt_cached_validation_quality*.json`；不据此在最终 test 后更换候选或调阈值。
- 仅对验证集 TensorRT 质量通过的冻结 `selective_2` 做同一确定性静态 bias 折叠（20 处），未使用最终 test 选择模型、修改量化层或阈值。派生图在 69 张 validation 与源图输出/决策完全一致；直接详细 TensorRT 引擎执行 20 个经层 datatype、权重、`i8i8` tactic 和运行 profile 交叉验证的 INT8 卷积。同一引擎 69 张 validation AUROC 0.97759、FNR 0、FPR 0.05，预设相对门槛通过；P5 按独立 TensorRT 实验范围通过。冷构建 51.589 s、热加载 0.0615 s。当时 C++ 在线程序仍是 ORT CUDA；后来已增加直接 TensorRT 后端，证据见本文件开头的追加记录。派生引擎仍没有独立最终 test 指标；不沿用源 ONNX 的最终测试结果。审计入口 `tools/verify_trt_route.py`，产物 `reports/trt_verified_selective_2.json`。
- TensorRT 默认允许 TF32 的 FP32 引擎与该 INT8 引擎使用同一验证图输入，启用传输，交替顺序做三轮各 1000 次测量：GPU 计算 P50 中位数 7.9565/2.7320 ms（2.912×），含传输的引擎迭代延迟 8.0444/2.8186 ms（2.854×）。这不是 C++ 决策延迟加速；原始计时与命令见 `reports/trt_engine_benchmark_with_transfers.json`。此前 `--noDataTransfers` 计时保留为历史探针，不作为实际输入传输证据。

官方依据：
- https://github.com/open-edge-platform/anomalib/tree/v2.1.0/src/anomalib/models/image/efficient_ad
- https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
- https://docs.nvidia.com/cuda/archive/12.5.1/cuda-installation-guide-microsoft-windows/index.html
- https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html
- https://learn.microsoft.com/windows/win32/api/synchapi/nf-synchapi-createwaitabletimerexw
- https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads
