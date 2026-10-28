# 本机演示

在 VS Code 的 PowerShell 终端进入项目目录，使用已校准的正式 FP32 模型和只含开发验证图片的回放清单：

```powershell
$run = 'runs/demo-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
.\scripts\run_windows.ps1 run --manifest artifacts/metal_nut/fp32/model_manifest.json --source data/replay.json --backend ort-cuda --postprocess cuda --fps 20 --output $run
Get-Content "$run/summary.json" -Encoding utf8
Get-Content "$run/events.jsonl" -Encoding utf8 -TotalCount 1
```

`summary.json` 展示完成数、错误/超时数和 P50/P95/P99；`events.jsonl` 为每帧的分数、OK/REVIEW 决策、独立状态与时间戳。REVIEW 帧的原始异常图 `map-*.tiff` 和预览 `heat-*.png` 写在同一运行目录。分数是异常图最大值，不是概率。此演示重复使用 validation 图片，不扩充质量测试样本。
