# 完整实验复现

本文保留项目原有的环境、训练、量化、审计与回放命令，作为 README 的进阶参考。数据、权重、引擎和运行目录需要本地生成；原始日志与大型 profile 不随源码分发。

正式训练、导出及最终测试请在独立工作副本执行，保留既有冻结记录。下列 `.venv-train` 指历史实测训练环境，准备方式见 [环境说明](ENVIRONMENT.md)。

## Local inputs

- MVTec AD `metal_nut` at `data/raw/metal_nut` ([source and noncommercial terms](https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads)).
- Official EfficientAD teacher at `pre_trained/efficientad_pretrained_weights/pretrained_teacher_small.pth` and Imagenette archive at `data/raw/imagenette2.tgz`. Their verified hashes are in `reports/downloads.json` and `reports/imagenette-verified.json`.
- ONNX Runtime GPU C++ 1.22.0 SDK at `third_party/onnxruntime-win-x64-gpu-1.22.0`; the build script accepts one nested copy of its directory.
- OpenCV 4.11.0 at `third_party/opencv` and TensorRT 10.9.0.34 at `third_party/TensorRT-10.9.0.34`.
- MSVC v143 14.44, Windows SDK, CUDA Toolkit 12.5. The actual paths and versions are written by `tools/environment.py`.

## Commands verified on this machine

Run from project root in PowerShell:

```powershell
.\.venv-tools\Scripts\python.exe tools/environment.py --output reports/environment-current.json
.\.venv-tools\Scripts\python.exe tools/prepare_data.py --config configs/metal_nut.yaml
.\scripts\probe_cuda.ps1
.\scripts\build_windows.ps1
.\.venv-tools\Scripts\python.exe tools/make_fixture.py
.\scripts\run_windows.ps1 run --manifest tests/fixtures/conv/model_manifest.json --source tests/fixtures/conv/replay.json --backend ort-cuda --postprocess cuda --fps 50 --deadline-ms 1000 --profile --output runs/p0-conv-cuda
.\.venv-train\Scripts\python.exe tools/train.py --config configs/metal_nut.yaml --smoke
.\.venv-train\Scripts\python.exe tools/export_onnx.py --config configs/metal_nut.yaml --smoke
```

Use a fresh `--output` directory for each run. The smoke checkpoint and synthetic fixture are explicitly excluded from final quality acceptance.

## Full model path

```powershell
.\.venv-train\Scripts\python.exe tools/train.py --config configs/metal_nut.yaml
.\.venv-train\Scripts\python.exe tools/export_onnx.py --config configs/metal_nut.yaml
.\.venv-train\Scripts\python.exe tools/verify_alignment.py --config configs/metal_nut.yaml --split validation --cpp
.\.venv-tools\Scripts\python.exe tools/evaluate.py --config configs/metal_nut.yaml --split validation
.\.venv-tools\Scripts\python.exe tools/quantize.py --config configs/metal_nut.yaml --phase base
.\.venv-tools\Scripts\python.exe tools/activation_diagnostics.py
.\.venv-tools\Scripts\python.exe tools/quantize.py --config configs/metal_nut.yaml --phase selective
.\.venv-tools\Scripts\python.exe tools/freeze_candidates.py --config configs/metal_nut.yaml
.\.venv-tools\Scripts\python.exe tools/pipeline.py --stage P7
.\.venv-tools\Scripts\python.exe tools/summarize_final_quality.py
```

The final `test` command checks `artifacts/frozen.json`; it rejects changed manifests, config or split hash. Do not use test results to revise candidates while claiming an unseen final evaluation. `tools/pipeline.py --config configs/metal_nut.yaml --stage P0|all` records commands, hashes and logs. It does not download resources.

For replay, use `scripts/run_windows.ps1 run --manifest artifacts/metal_nut/fp32/model_manifest.json --source data/replay.json --backend ort-cuda --fps 20 --output runs/replay`; add `--loop --duration-seconds 1800` for the soak. `benchmark` accepts `--mode model|pipeline|replay`, `--postprocess cpu|cuda`, `--warmup` and `--iterations`. `events.jsonl`, `summary.json` and the ORT profile are the raw evidence. `decision_latency_ms` starts at the scheduled arrival clock and includes queueing. CUDA Events are synchronized before reporting `model_gpu_ms`.

Exit codes: `0` expected completion, `2` configuration or model contract, `3` backend/dependency, `4` runtime infrastructure, `5` acceptance failure. A defect `REVIEW` is an expected completion. Model scores are raw maxima, not probabilities.

The completed 30-minute FP32 GPU replay is at `runs/soak-highres`; validation and resource records are `reports/soak-highres-validation.json` and `reports/soak-highres-memory-summary.json`. Run `.\.venv-tools\Scripts\python.exe tools/benchmark_candidates.py` for three same-backend precision rounds (100 warmups and 1000 measured iterations each); it currently exits nonzero because all three QDQ candidates require CPU EP fallback that the C++ CUDA backend explicitly disables. FP32 and FP16 complete, and every result or failure is recorded in `reports/precision_benchmarks.json` and `reports/precision-*.log`. QDQ generation, TensorRT provider assignment, and cached engine layer names do not establish executed INT8 precision; see `docs/ACCEPTANCE.md`.

The separate TensorRT INT8 experiment is reproducible from existing local files with these PowerShell commands. The original frozen model fails direct parsing; the derived graph is exploratory and is never substituted into the frozen final test.

```powershell
.\.venv-tools\Scripts\python.exe tools/fold_qdq_bias.py --candidate int8_percentile
.\.venv-tools\Scripts\python.exe tools/check_graph_equivalence.py --candidate int8_percentile
.\.venv-tools\Scripts\python.exe tools/build_trt_detailed.py --candidate int8_percentile_trt_compat
.\.venv-tools\Scripts\python.exe tools/run_trt_detailed.py --candidate int8_percentile_trt_compat
.\.venv-tools\Scripts\python.exe tools/verify_trt_detailed_runtime.py --candidate int8_percentile_trt_compat
```

The percentile detailed engine proves 21 INT8 convolutions were executed, but its output differs from ORT; `reports/trt_detailed_exploratory.json` retains both facts and the hashes. This candidate alone does not pass P5 or establish an INT8 speedup.

The frozen QDQ model's existing TensorRT Provider cache can be checked on validation only with `.\.venv-tools\Scripts\python.exe tools/evaluate_cached_trt.py --candidate int8_percentile`. It uses the frozen threshold and records scores, provider assignments and engine hash in `reports/trt_cached_validation_quality.json`; its FNR increase fails the preset quality gate. It does not modify the frozen model or test results.

The same validation check was run for the other two frozen QDQ candidates. To reproduce either one, first run `.\.venv-tools\Scripts\python.exe tools/verify_trt_int8.py --candidate int8_selective_2 --output reports/trt_int8_evidence_selective_2.json`, then `.\.venv-tools\Scripts\python.exe tools/evaluate_cached_trt.py --candidate int8_selective_2 --evidence reports/trt_int8_evidence_selective_2.json --output reports/trt_cached_validation_quality_selective_2.json`; substitute `selective_3` for `selective_2` for the third candidate. The first command currently exits nonzero because its cached engine has no per-layer precision metadata; the engine and its hash are still preserved for the second command. The `selective_2` TensorRT validation quality passes, while `selective_3` fails the FNR gate; neither has audited INT8 precision in its frozen engine.

P5 is satisfied by a hash-audited TensorRT experiment on a deterministic bias-folded copy of the already frozen `selective_2` candidate. The copy is identical to its source on all 69 validation images in ORT, and the direct TensorRT engine passes the preset validation quality gate with 20 actually executed INT8 convolutions. It has no independent final test result. Reproduce the engine with the existing local SDK:

```powershell
.\.venv-tools\Scripts\python.exe tools/fold_qdq_bias.py --candidate int8_selective_2
.\.venv-tools\Scripts\python.exe tools/check_graph_equivalence.py --candidate int8_selective_2
.\.venv-tools\Scripts\python.exe tools/build_trt_detailed.py --candidate int8_selective_2_trt_compat --precision int8
.\.venv-tools\Scripts\python.exe tools/run_trt_detailed.py --candidate int8_selective_2_trt_compat
.\.venv-tools\Scripts\python.exe tools/verify_trt_detailed_runtime.py --candidate int8_selective_2_trt_compat --output reports/trt_detailed_exploratory_selective_2.json
.\.venv-tools\Scripts\python.exe tools/evaluate_direct_trt.py --candidate int8_selective_2_trt_compat --output reports/trt_direct_validation_quality_selective_2.json
.\.venv-tools\Scripts\python.exe tools/build_trt_detailed.py --candidate fp32 --precision fp32
.\.venv-tools\Scripts\python.exe tools/benchmark_trt_engines.py --iterations 1000 --warmup-ms 1000
.\.venv-tools\Scripts\python.exe tools/verify_trt_route.py
```

`reports/trt_verified_selective_2.json` links the model, engine, layer profile, quality, cold/hot startup and raw timing evidence. With the same validation input and transfers enabled, the three-round TensorRT engine GPU-compute P50 ratio is 2.912× and iteration-latency P50 ratio is 2.854×; neither is a C++ decision-latency speedup. The previous no-transfer probe is retained separately. `tools/pipeline.py --stage P5` rechecks the recorded route.

Run the verified engine in the C++ detector, then audit all 69 validation events and the CPU/CUDA postprocessing comparison:

```powershell
.\scripts\build_windows.ps1
.\scripts\run_windows.ps1 run --manifest artifacts/metal_nut/int8_selective_2_trt_compat/model_manifest.json --source data/validation.json --backend trt-int8 --engine runs/trt-detailed-int8_selective_2_trt_compat/model.engine --engine-evidence reports/trt_verified_selective_2.json --postprocess cuda --fps 10 --deadline-ms 1000 --warmup 5 --output runs/trt-cpp-validation
.\scripts\run_windows.ps1 run --manifest artifacts/metal_nut/int8_selective_2_trt_compat/model_manifest.json --source data/validation.json --backend trt-int8 --engine runs/trt-detailed-int8_selective_2_trt_compat/model.engine --engine-evidence reports/trt_verified_selective_2.json --postprocess cpu --fps 10 --deadline-ms 1000 --warmup 5 --output runs/trt-cpp-validation-cpu-post
.\.venv-tools\Scripts\python.exe tools/validate_run.py --run runs/trt-cpp-validation --output reports/trt-cpp-validation-schema.json
.\.venv-tools\Scripts\python.exe tools/audit_trt_cpp.py
```

Use fresh output directories if repeating. `trt-int8` checks model, engine and evidence hashes before deserialization and rejects a mismatched engine. The C++ path produced 69/69 valid events, matched direct TensorRT scores within `3.34e-6`, and reached the same validation AUROC `0.97759`, FNR `0`, FPR `0.05`; the audit is `reports/trt-cpp-validation-quality.json`. The existing frozen `test` set is deliberately not reused as an independent final test for this derived engine. Engine-level timing is separate from C++ decision timing.

The 30-minute INT8 replay has also completed with 36001/36001 frames and decision P95 `32.5113 ms`. Reproduce it with the same `run` command above, replacing the scheduling/output arguments with `--fps 20 --loop --duration-seconds 1800 --deadline-ms 50 --warmup 20 --output runs/trt-cpp-soak`; then run `tools/validate_run.py --run runs/trt-cpp-soak --output reports/trt-cpp-soak-schema.json` and `tools/audit_trt_soak.py`. Results and limits are in `docs/ACCEPTANCE.md`.

For the same-backend C++ FP32/INT8 timing comparison, run `.\.venv-tools\Scripts\python.exe tools/benchmark_trt_cpp.py --iterations 1000 --warmup 100`, followed by `.\.venv-tools\Scripts\python.exe tools/audit_trt_cpp_benchmark.py`. This uses the existing `trt-fp32` engine only as a timing comparator: its validation FNR at the frozen ORT threshold is `0.03448`, outside the specified relative quality limit. The measured C++ pipeline wall P50 ratio is `2.619×`; the benchmark omits scheduling and decode, so it does not establish a decision-latency speedup. See `reports/trt-cpp-benchmark-audit.json`.
