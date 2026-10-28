param([string]$QtRoot = 'D:/anaconda/Library', [string[]]$GuiArgs = @())
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$qtBin = Join-Path $QtRoot 'bin'
$qtPlugins = Join-Path $QtRoot 'plugins'
if (!(Test-Path (Join-Path $qtBin 'Qt5Widgets_conda.dll'))) {
    Write-Error "Qt Widgets runtime missing at $qtBin. Pass -QtRoot for an installed Qt 5.15 SDK."
    exit 3
}
if (!(Test-Path (Join-Path $projectRoot 'build/inspection_gui.exe'))) {
    Write-Error 'GUI executable missing. Run scripts/build_windows.ps1 -BuildGui first.'
    exit 3
}
$cudaRoot = $env:CUDA_PATH
if (!$cudaRoot) { $cudaRoot = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5' }
$dllDirs = @($qtBin, "$projectRoot/third_party/opencv/build/x64/vc16/bin", "$projectRoot/third_party/TensorRT-10.9.0.34/lib", "$cudaRoot/bin")
$nvidiaRoot = 'C:/Users/yuxiangw/ort-env/Lib/site-packages/nvidia'
if (Test-Path $nvidiaRoot) { $dllDirs += Get-ChildItem -LiteralPath $nvidiaRoot -Directory | ForEach-Object { Join-Path $_.FullName 'bin' } }
$env:PATH = ($dllDirs -join ';') + ';' + $env:PATH
$env:QT_QPA_PLATFORM_PLUGIN_PATH = Join-Path $qtPlugins 'platforms'
Set-Location $projectRoot
$launch = @{ FilePath = "$projectRoot/build/inspection_gui.exe"; WorkingDirectory = $projectRoot; PassThru = $true; Wait = $true }
if ($GuiArgs.Count) { $launch.ArgumentList = $GuiArgs }
$guiProcess = Start-Process @launch
exit $guiProcess.ExitCode
