param([string]$OrtRoot = "$PSScriptRoot/../third_party/onnxruntime-win-x64-gpu-1.22.0", [string]$OpenCV_DIR = $env:OpenCV_DIR, [switch]$BuildGui, [string]$QtRoot = 'D:/anaconda/Library')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
if (!$OpenCV_DIR) { $OpenCV_DIR = Join-Path $projectRoot 'third_party/opencv/build/x64/vc16/lib' }
Set-Location $projectRoot
if (!(Test-Path "$OrtRoot/include/onnxruntime_cxx_api.h")) {
    $nested = Join-Path $OrtRoot 'onnxruntime-win-x64-gpu-1.22.0'
    if (Test-Path "$nested/include/onnxruntime_cxx_api.h") { $OrtRoot = $nested }
}
$vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs -or !(Test-Path "$vs/Common7/Tools/VsDevCmd.bat")) {
    Write-Error 'BLOCKED: usable MSVC x64 and Windows SDK missing. Repair/install VS 2022 Desktop development with C++; see docs/ENVIRONMENT.md.'
    exit 3
}
# Import developer environment; no global PATH mutation.
cmd /s /c "`"$vs\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 -vcvars_ver=14.44 >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$env:PATH = "$projectRoot\.venv-tools\Scripts;$projectRoot\.venv\Scripts;$projectRoot\third_party\opencv\build\x64\vc16\bin;$env:PATH"
$env:VSLANG = '1033'
$commands = @(
  @('cmake','--fresh','--preset','windows-release',"-DORT_ROOT=$OrtRoot", "-DOpenCV_DIR=$OpenCV_DIR"),
  @('cmake','--build','--preset','windows-release'),
  @('ctest','--preset','windows-release')
)
if ($BuildGui) {
    if (!(Test-Path "$QtRoot/lib/cmake/Qt5Widgets/Qt5WidgetsConfig.cmake")) {
        Write-Error "Qt 5.15 Widgets CMake SDK missing at $QtRoot. Place a user-installed MSVC-compatible Qt SDK there or pass -QtRoot."
        exit 3
    }
    $commands[0] = @($commands[0]) + @('-DINSPECTION_QT_UI=ON', "-DCMAKE_PREFIX_PATH=$QtRoot")
}
foreach($command in $commands) {
    Write-Output ($command -join ' ')
    $ErrorActionPreference = 'Continue'
    & $command[0] $command[1..($command.Length-1)]
    if($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
