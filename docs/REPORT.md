# 可复现研究原型报告（2026-09-30）

本报告只描述本机实测。原始数据、模型、命令、进程日志与哈希见 `reports/`、`runs/`、`artifacts/`；运行入口见仓库 `README.md`。项目数据为 MVTec AD `metal_nut` 单类别，最终质量集合有 79 张独立图片。循环回放只用于稳定性，不计入质量样本数。

## 环境与训练

`reports/environment-install-check.json` 记录了 RTX 4060 Laptop GPU、CUDA Toolkit 12.5.40、MSVC v143 14.44、Windows SDK、OpenCV 4.11.0、ORT GPU 1.22.0 与 TensorRT 10.9.0.34 的本地路径。`.venv-train` 使用 Anomalib 2.1.0、torch 2.4.1+cu124；`.venv-tools` 承担离线量化。所有下载由用户提供，执行脚本没有自动下载依赖。

固定清单数量为 train_normal 154、stats_normal 33、dev_normal 33、calibration 128、validation 69、test 79；清单和逐文件 hash 在 `reports/data_protocol.json` 等文件。正式 EfficientAD-S GPU 训练运行 10,000 步，权重 SHA-256 为 `bf15f84a0f2e267a28fe9a43d98518c9795c88fe83471a149083079140fdf7a4`，模型恢复与重复推理差异均为 0。证据为 `artifacts/metal_nut/training/training.json` 与 `reports/train-full.log`。

## 导出、C++ CUDA 与品质

正式 FP32 ONNX 的 checker 通过。69 张验证图的 PyTorch/ORT 相同张量最大绝对差为 `2.3842e-6`；Python ORT 与 C++ 原图全链路分数最大差为 `1.7882e-7`，在预设 `atol=1e-5, rtol=1e-4` 内。`reports/alignment.json` 保存逐图值。正式 C++ ORT profile 的 3,000 个节点事件均分配到 CUDAExecutionProvider，CPU 节点为 0，见 `reports/fp32-provider-evidence.json`。

FP32 验证集 69/69 有效，AUROC `0.97931`、FNR `0`、FPR `0.05`；20 FPS 单次 C++ 回放 69/69 完成，错误/过载/超时为 0，决策延迟 P95 `37.1723 ms`。原始结果为 `artifacts/metal_nut/fp32/validation_quality.json` 与 `runs/baseline/summary.json`，不能代替 30 分钟稳定性验收。

## 量化与实际精度

FP16、MinMax/Percentile QDQ 和三组依据 42 个匹配激活张量误差排序的选择性排除候选，均完成验证集评测。MinMax 和 selective_1 没有达到预先规定的 AUROC/FNR 门槛；percentile、selective_2/3 达到验证门槛。逐候选图、精度意图、校准清单及哈希在 `artifacts/metal_nut/*/precision_intent.json` 与 `reports/quantization.json`；逐候选品质在各目录的 `validation_quality.json`。QDQ 节点和文件名只表示精度意图。

ORT TensorRT Provider 实际执行了 QDQ 主子图，初始 percentile profile 中 CPU 回退有 22 个小张量 DequantizeLinear，没有 CPU Conv/MatMul。其缓存 engine 的 Inspector 仅导出 61 个层名，缺少构建时的精度元数据；**该缓存引擎本身不能证明 INT8 层精度**。`reports/trt_int8_evidence.json` 指向模型 hash、profile、缓存 engine hash 和原始 Inspector 文件。NVIDIA [Engine Inspector 文档](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html)说明，详细程度由构建时 profiling verbosity 决定。后续 P5 通过依据是独立构建并实测的详细 TensorRT 引擎，而非此缓存引擎。

三个冻结 QDQ 候选各自在缓存 TensorRT 引擎上按冻结阈值实际运行 69 张 validation，均 69/69 有效、无 ERROR；没有重新校准阈值，也没有访问最终 test。`percentile` 的 AUROC/FNR/FPR 为 `0.97845/0.03448/0.075`，相对同模型 ORT CUDA 路线 FNR 从 `0` 增至 `0.03448`，超过最大增量 `0.02`，配对分数最大差 `0.13658`。`selective_2` 为 `0.97672/0/0.05`，验证集质量通过；`selective_3` 为 `0.97672/0.06897/0.05`，FNR 失败。逐图分数、各自引擎 hash、profile 和计数在 `reports/trt_cached_validation_quality*.json` 与 `runs/trt-cached-validation-*`。这些缓存引擎仍缺可审计层精度，因此只用于候选的 TensorRT 验证集质量比较。

为进一步隔离问题，本地 `trtexec` 对冻结的 `int8_percentile` ONNX 直接构建失败，错误在静态 bias 的 `DequantizeLinear`，原始日志为 `runs/trt-detailed-int8_percentile/build.log`。随后只在独立的探索性副本中将 22 个静态 bias DQ 折叠为 float32 常量；源图不变，69 张 validation 的输出逐张最大差为 0，见 `artifacts/metal_nut/int8_percentile_trt_compat/equivalence.json`。该副本的直接 TensorRT 引擎确实运行；详细 Inspector 元数据中 21 个卷积层有 INT8 权重、INT8 输入和 `i8i8` tactic，并与实际运行 profile 的层名/耗时匹配。完整证据在 `reports/trt_detailed_exploratory.json`。但直接 TensorRT 对首张验证图的输出相对 ORT 最大差为 `0.17016`，未对该 percentile 引擎作合格质量声明。

对验证集质量通过的冻结 `int8_selective_2` 候选，采用同一机械式 bias 常量折叠得到独立 TensorRT 副本；20 个 bias DQ 被折叠，源图与副本在 69 张 validation 上输出最大差 `0`、决策一致 `69/69`，阈值不变。其直接 TensorRT 详细引擎的 20 个卷积层具有 INT8 权重、INT8 输入、`i8i8` tactic，并与实际 runtime profile 的层名和非零耗时逐层匹配；直接引擎没有 ORT CPU EP 回退。该**同一引擎**对 69 张 validation 产生 69/69 有效分数，AUROC `0.97759`、FNR `0`、FPR `0.05`，通过预设相对质量门槛。完整模型 hash、engine hash、逐图输入/输出/log hash 和门槛核验在 `reports/trt_verified_selective_2.json`、`reports/trt_detailed_exploratory_selective_2.json` 与 `reports/trt_direct_validation_quality_selective_2.json`。冷引擎构建 `51.589 s`，热反序列化 `0.0615 s`，分开报告。这个派生引擎没有使用最终 test；不能借冻结源 ONNX 的最终测试指标为其作新声明。C++ 在线程序现提供 `trt-int8` 后端；69 张验证集完整事件审计见 `reports/trt-cpp-validation-quality.json`，对直接 TensorRT 分数最大差 `3.34e-6`、决策 `69/69` 一致、CUDA/CPU 后处理分数完全一致。错误 FP32 引擎被 hash 校验拒绝，日志见 `reports/trt-cpp-wrong-engine.log`。

同一 TensorRT 后端、同一张验证图输入，启用主机/设备传输的三轮各 1000 次对照：FP32 与 INT8 的 GPU 计算 P50 中位数分别为 `7.9565/2.7320 ms`（`2.912×`），含传输的引擎迭代延迟 P50 分别为 `8.0444/2.8186 ms`（`2.854×`）。原始 6000 次计时、各轮 P95/P99、引擎 hash 和命令见 `reports/trt_engine_benchmark_with_transfers.json` 与 `runs/trt-engine-benchmark-with-transfers`。FP32 引擎保留 TensorRT 默认可用的 TF32，未传 `--noTF32`。此前关闭传输的纯内核探针保留在 `reports/trt_engine_benchmark.json`，不用于输入数据已被传到设备的声明。这证明本机引擎层面测到较短计算和迭代时间，**不代表** C++ 程序的决策延迟或整管线获得同等提速。

C++ 程序同样加入 `trt-fp32` 作为同一 TensorRT 后端的计时对照。交替顺序三轮、每轮 100 次预热加 1000 次测量、同一验证图，在含预处理、传输、推理和 CUDA 后处理的 pipeline 模式中，FP32/INT8 wall P50 三轮中位数为 `8.3386/3.1841 ms`（`2.619×`），模型 GPU P50 为 `7.8099/2.7351 ms`（`2.855×`）。原始 6000 条计时和 hash 在 `reports/trt_cpp_pipeline_benchmark.json`、`runs/trt-cpp-pipeline-benchmark`，交叉核验与品质限制在 `reports/trt-cpp-benchmark-audit.json`。pipeline benchmark 不包含回放队列、调度或解码，因此不声明决策延迟加速。FP32 TensorRT 引擎虽然 69/69 SUCCESS，但沿用正式 ORT 阈值时出现 1 个漏检，FNR `0.03448` 超过相对基线最大增量 `0.02`，见 `reports/trt-cpp-fp32-validation-quality.json`；它只作性能对照，不能称为另一条质量合格的部署路线。INT8 后端验证集通过不替代其独立最终 test。

## CUDA 后处理与性能

随机长度、阈值边界、NaN/Inf、掩膜的 CPU/CUDA 对照通过 CTest；正式模型的 69 帧验证回放中，两种后处理的分数完全相同，状态和决策 69/69 相同。100 次预热、1,000 次测量、各三轮的后处理中位数为 CPU `0.2446 ms`、CUDA `0.05069 ms`。整管线中位数分别为 `13.2163 ms`、`13.3254 ms`；按 CPU 三轮后 CUDA 三轮的次序，GPU 温度从 53°C 升到 87°C。交替顺序补测说明整体差异很小且随模型 GPU 时间变化，不能将全部差异归因于 CUDA kernel。`reports/postprocess_ab.json`、`runs/bench-*` 与温度观测文件保存原始依据。

## 冻结后的最终测试

`artifacts/frozen.json` 在读取 test 品质前固定了 5 个验证达标候选、阈值、配置和清单 hash。最终测试每个候选只计 79 张独立图；汇总在 `reports/final_quality.json`。FP32 与 FP16 均为 TP 57、FP 0、TN 15、FN 7，AUROC `0.97604`。int8_percentile 的 AUROC `0.95521`，相对 FP32 下降 `0.02083`，**不满足** `≤0.01` 门槛。int8_selective_2/3 的 AUROC `0.98958`，FN 8，FNR 相对增加 `0.01562`，在既定门槛内；但其实际 INT8 执行仍未验证。每个模型的漏检图片路径、分数及冻结阈值列于 `reports/failure_samples.json`，供事后分析；不据此重新筛层或调阈值。像素阈值未标定，像素质量指标保持未定义。

最终测试的正常图仅 15 张、异常图 64 张。`reports/final_quality.json` 还给出按独立图片计算的 Wilson 95% 比例区间；例如 FP32 FNR 的区间为 `[0.0540, 0.2090]`，FPR 的区间为 `[0, 0.2039]`。这反映小样本不确定性，不用 30 分钟重复回放缩窄区间。

完整 30 分钟复测 `runs/soak-highres` 已通过：20 FPS、36001/36001 帧完成，实测 1800.026 秒；0 推理错误、0 队列溢出、1 超时（0.00278%）；决策延迟 P50 `25.8031 ms`、P95 `26.7411 ms`、P99 `27.2819 ms`，满足既定 P95 ≤40 ms 和超时率 ≤1% 门槛。命令见 `docs/STATUS.md` P7，原始 `events.jsonl`、`summary.json` 和 `reports/soak-highres-validation.json` 证明 36001 个事件均满足 schema、唯一终态及计数。资源采样 `reports/soak-highres-memory-summary.json` 记录 53 次进程工作集，范围 832.19–834.79 MiB，首末为 833.57/833.38 MiB；WDDM 无进程级显存数据，不能宣称测得精确进程显存。首次未通过的回放保留在 `runs/soak`。

C++ 直接 TensorRT INT8 后端另做同样 20 FPS、1800 秒循环回放：`runs/trt-cpp-soak` 完成 36001/36001 帧、1800.025 秒，0 推理错误和队列溢出、5 超时（0.0139%），决策 P95 `32.5113 ms`，通过既定时延门槛。`reports/trt-cpp-soak-schema.json` 和 `reports/trt-cpp-soak-audit.json` 审计所有事件的 schema、模型/引擎身份和终态；人工 1000 FPS 过载实验产生 66 个队列溢出，均进入复检，见 `reports/trt-cpp-overload-schema.json`。该次 30 分钟回放未采样进程工作集；另做 300.022 秒、6001/6001 帧的 INT8 资源补测，27 次采样覆盖 271.203 秒，工作集范围 295.68–298.17 MiB、首末 +1.31 MiB，设备全局显存首末 +1 MiB，见 `reports/trt-cpp-memory-300s-summary.json`。这不代表 30 分钟内存曲线；WDDM 下进程级 VRAM 不可得。循环帧不计为新的品质样本。

同一 C++ ORT CUDA 后端的三轮模型基准（每轮 100 次预热、1000 次测量）见 `reports/precision_benchmarks.json`：FP32 的 wall P50 三轮中位数 `12.854 ms`，FP16 为 `12.9906 ms`；model GPU P50 分别为 `12.6258 ms`、`12.7582 ms`。在这组负载上 FP16 没有可验证的提速。三个冻结 QDQ 候选在 C++ ORT 后端严格禁止 CPU EP 回退时均无法初始化，失败命令和日志在 `reports/precision-int8_*.log`。项目的 FP32 C++ ORT CUDA 主线与直接 TensorRT INT8 路线均已可运行；C++ 同后端 pipeline 计时对照已完成，但不是含调度、队列和解码的决策延迟加速证据，且 FP32 TensorRT 计时对照未过质量门槛。派生 INT8 引擎的独立最终质量测试也未验证。
