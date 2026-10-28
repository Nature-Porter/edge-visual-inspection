"""Shared offline contracts. Paths in config are project-relative."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_DLL_HANDLES = []
_CUDA_DLL_PATH_READY = False

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))

def verified_records(split):
    doc = read(ROOT / 'data' / (split + '.json'))
    for r in doc['records']:
        for key, hashkey in [('path','sha256'), ('mask_path','mask_sha256')]:
            if r.get(key) and sha256(ROOT / r[key]) != r[hashkey]:
                raise ValueError(f'changed data: {r[key]}')
    return doc['records']

def preprocess(path):
    import cv2
    import numpy as np
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or not image.size:
        raise ValueError(f'INVALID_IMAGE: {path}')
    image = cv2.cvtColor(cv2.resize(image, (256,256), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(image.transpose(2,0,1)[None], dtype=np.float32) / np.float32(255)

def ort_session(path, profile=None, provider='CUDAExecutionProvider'):
    global _CUDA_DLL_PATH_READY
    if os.name == 'nt' and provider != 'CPUExecutionProvider' and not _CUDA_DLL_PATH_READY:
        # The lightweight quantization venv has no torch package to preload CUDA DLLs.
        # Reuse the already-installed runtime without modifying global PATH or copying DLLs.
        nvidia_root = Path(os.environ.get('INSPECTION_NVIDIA_DLL_ROOT', Path.home()/'ort-env/Lib/site-packages/nvidia'))
        dll_dirs = [Path(os.environ.get('CUDA_PATH', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5'))/'bin']
        if nvidia_root.is_dir():
            dll_dirs.extend(p/'bin' for p in nvidia_root.iterdir() if p.is_dir())
        available = [directory for directory in dll_dirs if directory.is_dir()]
        for directory in available:
            _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
        # cuDNN loads some sublibraries by basename through the process PATH.
        os.environ['PATH'] = os.pathsep.join(map(str, available)) + os.pathsep + os.environ['PATH']
        _CUDA_DLL_PATH_READY = True
    import onnxruntime as ort
    if hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
    options = ort.SessionOptions()
    if profile:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile)
    providers = [(provider, {'use_tf32':'0'})] if provider == 'CUDAExecutionProvider' else [provider]
    session = ort.InferenceSession(str(path), options, providers=providers)
    session.disable_fallback()
    if provider not in session.get_providers():
        raise RuntimeError(f'requested {provider} unavailable (no silent CPU fallback)')
    return session
