$ErrorActionPreference = 'Continue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
$vs = & $vswhere -latest -products '*' -property installationPath
cmd /s /c "`"$vs\Common7\Tools\VsDevCmd.bat`" -arch=x64 -host_arch=x64 -vcvars_ver=14.44 >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
New-Item -ItemType Directory -Force build-probes | Out-Null
& nvcc -std=c++17 -arch=sm_89 tests/unit/cuda_smoke.cu -o build-probes/cuda_smoke.exe
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& ./build-probes/cuda_smoke.exe
exit $LASTEXITCODE
