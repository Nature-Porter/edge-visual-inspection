# 独立验收

| 项目 | 状态 | 证据 |
|---|---|---|
| code_complete | passed | 规格 P0–P8 的代码、构建、训练、运行、验证集 INT8 实验与报告均有实际通过证据；C++ 检测程序现支持 hash 审计的 TensorRT INT8 后端，69 张验证集全链路通过。正式 FP32 基线仍为 ORT CUDA。见 docs/STATUS.md、reports/pipeline_states.json、reports/trt-cpp-validation-quality.json |
| gpu_verified | passed | 正式 FP32 C++ ORT profile 3000 个节点事件全部由 CUDAExecutionProvider 执行、CPU 节点 0，见 reports/fp32-provider-evidence.json 与 runs/fp32-provider-profile；69/69 真实验证回放完成 |
| quality_pass | partial | 冻结后 79 张独立 test：FP32/FP16/selective_2/3 在原 ORT CUDA 路线达相对品质门槛；int8_percentile AUROC 降 0.02083，未达 ≤0.01，见 reports/final_quality.json。C++ TensorRT INT8 `selective_2` 在 69 张 validation 上 AUROC 0.97759、FNR 0、FPR 0.05，预设相对门槛通过；该派生引擎未使用独立最终 test，不能沿用源 ONNX 的最终测试结论。见 reports/trt-cpp-validation-quality.json |
| latency_pass | passed | 正式 FP32 C++ ORT CUDA、20 FPS、1800.026 秒回放：决策 P95 26.7411 ms，低于 40 ms；36001 帧中 1 次超时（0.00278%），见 runs/soak-highres/summary.json 与 reports/soak-highres-validation.json。首次未通过的结果保留在 runs/soak |
| stability_pass | passed | 36001/36001 帧完成、0 错误/队列溢出、1 次超时，事件 schema 和唯一终态 36001/36001 通过；进程工作集 53 次采样为 832.19–834.79 MiB，无观察到持续增长。见 runs/soak-highres/summary.json、reports/soak-highres-validation.json、reports/soak-highres-memory-summary.json；WDDM 下无进程级 VRAM 数值 |
| int8_execution_verified | passed | TensorRT 10.9 引擎执行了 20 个 INT8 卷积：详细层的 INT8 权重/输入、`i8i8` tactic 与非零运行 profile 对应；同一引擎现由 C++ 程序直接加载，验证集 69/69 SUCCESS、质量门槛通过，模型/引擎/事件 hash 见 reports/trt_verified_selective_2.json 和 reports/trt-cpp-validation-quality.json。尚无经同等条件核验的 C++ 整管线提速声明 |
| cpp_int8_replay | passed | C++ 直接 TensorRT INT8，20 FPS、1800.025 秒：36001/36001 帧、0 错误/队列溢出、5 超时（0.0139%）、决策 P95 32.5113 ms；全部事件 schema、引擎身份和时间门槛审计通过，见 runs/trt-cpp-soak/summary.json、reports/trt-cpp-soak-audit.json。66 次人工过载均归为复检，见 reports/trt-cpp-overload-schema.json。独立 300 秒补测 6001/6001 帧，27 次工作集采样范围 295.68–298.17 MiB、首末 +1.31 MiB，见 reports/trt-cpp-memory-300s-summary.json；不能转称 30 分钟内存曲线，进程 VRAM 不可得。循环回放不增加品质样本数 |
| cpp_int8_speed_comparison | measured, limited | 同一 C++ TensorRT 后端、三轮各 1000 次、同一输入的 pipeline 调用 wall P50 中位数 FP32 8.3386 ms、INT8 3.1841 ms（2.619×），模型 GPU P50 7.8099/2.7351 ms（2.855×），见 reports/trt-cpp-benchmark-audit.json。此模式不含调度/队列/解码，不能称决策延迟加速；FP32 TensorRT 对照引擎沿用原阈值的 FNR 0.03448 未过质量门槛，见 reports/trt-cpp-fp32-validation-quality.json，因此也不能宣称两条合格部署路线的性能差 |
| qt_frontend | passed, fixture scope | 可选 Qt 5.15 C++ GUI 已在本机 MSVC 构建成功，原 CTest 2/2 通过；GUI 通过 QProcess 启动 C++ ORT CUDA fixture、100/100 事件 schema 校验通过，见 reports/gui-build.log、reports/gui-preview.png、reports/gui-smoke-validation.json。界面不替代正式模型质量/GPU/INT8 证据，离线训练命令不自动执行。 |

按任务书规定的研究原型交付范围，P0–P8 的实现、构建和阶段验证已完成，原定冻结后 `test.json` 的 79 张最终测试也已执行；不需要第二批数据才能满足原定的一次测试流程。`quality_pass` 保持 partial，因为冻结候选 `int8_percentile` 未达品质门槛，且最终测试后新增的直接 TensorRT INT8 派生引擎只有验证集品质证据。若要为该派生引擎另作独立最终品质声明，才需要一批真正未参与开发和此前最终测试的新图；不能重复使用原 `test.json`。
