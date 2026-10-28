"""Read-only inventory; distinguishes installed tools from PATH visibility."""
import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from common import ROOT, save, sha256

def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, errors='replace', timeout=45)
        return {'argv':argv, 'returncode':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'argv':argv, 'error':str(e)}

def inventory():
    vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    vs = command([str(vswhere), '-all', '-products', '*', '-format', 'json'])
    import json
    installations = json.loads(vs.get('stdout') or '[]')
    for item in installations:
        root = Path(item['installationPath'])
        item['path_exists'] = root.exists()
        item['cl_files'] = [str(p) for p in root.glob('VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe')]
        item['devcmd_exists'] = (root/'Common7/Tools/VsDevCmd.bat').exists()
    cuda = os.environ.get('CUDA_PATH', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5')
    sdk_root='C:/Program Files (x86)/Windows Kits/10'
    if sys.platform=='win32':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SOFTWARE\Microsoft\Windows Kits\Installed Roots') as key:
                sdk_root=winreg.QueryValueEx(key,'KitsRoot10')[0]
        except OSError:pass
    return {'platform':platform.platform(), 'python':sys.version, 'executable':sys.executable,
            'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
            'path_tools':{t:shutil.which(t) for t in ['cl','cmake','ninja','nvcc','git','nvidia-smi']},
            'cuda_toolkit_path':cuda, 'nvcc':command([str(Path(cuda)/'bin/nvcc.exe'),'--version']),
            'gpu':command(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw','--format=csv']),
            'visual_studio':installations,
            'windows_sdk_root':sdk_root,'windows_sdk_include':list(map(str,(Path(sdk_root)/'Include').glob('*'))),
            'old_ort_environment':command(['C:/Users/yuxiangw/ort-env/Scripts/python.exe','-c','import sys,onnxruntime as o; print(sys.executable,o.__version__,o.get_available_providers())']),
            'spec_sha256':sha256(next(ROOT.parent.glob('*Codex开发任务书.md')))}

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',default='reports/environment.json'); a=p.parse_args()
    result=inventory(); save(a.output,result)
    print(f'Inventory saved: {a.output}; usable MSVC: {any(v["cl_files"] for v in result["visual_studio"])}')
