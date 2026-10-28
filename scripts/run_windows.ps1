param([Parameter(ValueFromRemainingArguments=$true)][string[]]$InspectionArgs)
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$cudaRoot = $env:CUDA_PATH
if (!$cudaRoot) { $cudaRoot = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5' }
$dllDirs = @("$projectRoot/third_party/opencv/build/x64/vc16/bin", "$projectRoot/third_party/TensorRT-10.9.0.34/lib", "$cudaRoot/bin")
# Existing CUDA/cuDNN redistributables are read-only; do not copy or replace installations.
$nvidiaRoot = 'C:/Users/yuxiangw/ort-env/Lib/site-packages/nvidia'
if(Test-Path $nvidiaRoot) { $dllDirs += Get-ChildItem -LiteralPath $nvidiaRoot -Directory | ForEach-Object { Join-Path $_.FullName 'bin' } }
$env:PATH = ($dllDirs -join ';') + ';' + $env:PATH
& "$projectRoot/build/inspection.exe" @InspectionArgs
exit $LASTEXITCODE
