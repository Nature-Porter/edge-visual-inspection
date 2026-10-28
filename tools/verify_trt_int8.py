"""Verify target ORT TensorRT subgraph and inspect its cached engine layers.

No provider-list or QDQ-only claims; every successful assertion uses raw evidence.
"""
import argparse
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from common import ROOT, config, read, save, sha256, preprocess, verified_records

def run(config_path,candidate,pointer_output='reports/trt_int8_evidence.json'):
    cfg=config(config_path)
    import numpy as np
    import onnx
    import onnxruntime as ort
    from integration_tests import runtime_env
    package=ROOT/'artifacts/metal_nut'/candidate
    manifest=read(package/'model_manifest.json')
    model=package/manifest['onnx_path']
    if manifest['requested_precision'] not in ('INT8/FP32','INT8/FP16') or sha256(model)!=manifest['onnx_sha256']:
        raise ValueError('quantized target contract/hash mismatch')
    graph=onnx.load(model)
    if not any(n.op_type=='QuantizeLinear' for n in graph.graph.node) or not any(n.op_type=='DequantizeLinear' for n in graph.graph.node):
        raise ValueError('target model lacks QDQ nodes')
    trt=ROOT/'third_party/TensorRT-10.9.0.34'
    executable=trt/'bin/trtexec.exe'
    if not executable.is_file():raise FileNotFoundError(f'user-managed TensorRT SDK missing: {executable}')
    env=runtime_env()
    env['PATH']=str(trt/'lib')+';'+str(trt/'bin')+';'+env['PATH']
    # ORT loads the provider in this process, and TensorRT loads dependencies by basename.
    os.environ['PATH']=env['PATH']
    handles=[]
    if hasattr(os,'add_dll_directory'):
        for folder in env['PATH'].split(';'):
            if Path(folder).is_dir():
                try:handles.append(os.add_dll_directory(folder))
                except OSError:pass
    gpu='unknown'
    try:gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    except Exception:pass
    run_dir=ROOT/'runs'/('trt-int8-'+candidate+'-'+str(time.time_ns()));run_dir.mkdir(parents=True)
    key={'model_sha256':sha256(model),'input_shape':manifest['input']['shape'],'output_shape':manifest['output']['shape'],
      'ort_version':ort.__version__,'trt_version':'10.9.0.34','gpu':gpu,'requested_precision':manifest['requested_precision'],
      'config_sha256':sha256(config_path)}
    cache=run_dir/'engine_cache';cache.mkdir()
    options=ort.SessionOptions();options.enable_profiling=True;options.profile_file_prefix=str(run_dir/'ort-profile')
    providers=[('TensorrtExecutionProvider',{'trt_engine_cache_enable':True,'trt_engine_cache_path':str(cache),'trt_detailed_build_log':True}),
        ('CUDAExecutionProvider',{'use_tf32':False}),'CPUExecutionProvider']
    report={'status':'blocked','int8_execution_verified':False,'candidate':candidate,'key':key,'ort_providers_requested':providers,
        'profile':None,'provider_counts':None,'engine_inspection':None,'validation_quality':None,'reason':None}
    save(run_dir/'cache_key.json',key)
    try:
        start=time.perf_counter();session=ort.InferenceSession(str(model),options,providers=providers)
        report['cold_session_seconds']=time.perf_counter()-start
        session.disable_fallback()
        sample=verified_records('validation')[0]
        y=session.run([manifest['output']['name']],{manifest['input']['name']:preprocess(ROOT/sample['path'])})[0]
        if list(y.shape)!=manifest['output']['shape'] or not np.isfinite(y).all():raise ValueError('invalid TRT result')
        profile=session.end_profiling();report['profile']=profile;report['profile_sha256']=sha256(profile)
        nodes=[e.get('args',{}) for e in read(profile) if e.get('cat')=='Node' and e.get('args',{}).get('provider')]
        report['provider_counts']={f'{p}/{o}':n for (p,o),n in Counter((n['provider'],n.get('op_name','')) for n in nodes).items()}
        report['cpu_heavy_fallback']=[n for n in nodes if n['provider']=='CPUExecutionProvider' and n.get('op_name') in ('Conv','MatMul','Gemm','QLinearConv','QLinearMatMul')]
        report['cpu_control_fallback']=[n for n in nodes if n['provider']=='CPUExecutionProvider' and n not in report['cpu_heavy_fallback']]
        if not any(n['provider']=='TensorrtExecutionProvider' for n in nodes):raise RuntimeError('target had no TensorRT execution events')
        engines=list(cache.rglob('*.engine'))
        report['cached_engines']=[{'path':str(p),'sha256':sha256(p)} for p in engines]
        if not engines:raise RuntimeError('no ORT TensorRT cached engine to inspect')
        layer_path=run_dir/'trt_layers.json'
        cmd=[str(executable),f'--loadEngine={engines[0]}','--skipInference','--dumpLayerInfo','--profilingVerbosity=detailed',f'--exportLayerInfo={layer_path}']
        completed=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,errors='replace',timeout=180)
        (run_dir/'trtexec.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
        report['trtexec_command']=cmd;report['trtexec_exit_code']=completed.returncode
        if completed.returncode or not layer_path.exists():raise RuntimeError('trtexec cached-engine layer inspection failed')
        layers=read(layer_path);entries=layers.get('Layers',layers) if isinstance(layers,dict) else layers
        details=[]
        for layer in entries:
            if not isinstance(layer,dict):continue
            name=str(layer.get('Name',''));kind=str(layer.get('LayerType',''))
            precision=str(layer.get('Precision',''));tactic=str(layer.get('TacticName',''))
            compute=any(op in (name+' '+kind).lower() for op in ('conv','matmul','gemm','matrixmultiply'))
            int8_compute=compute and ('int8' in precision.lower() or 'int8' in tactic.lower() or 'i8i8' in tactic.lower())
            details.append({'name':layer.get('Name'),'precision':layer.get('Precision'),'layer_type':layer.get('LayerType'),
                'int8_marker':int8_compute,'raw':layer})
        int8=[d for d in details if d['int8_marker']]
        report['engine_inspection']={'file':str(layer_path),'sha256':sha256(layer_path),'layer_count':len(entries),
            'metadata_level':'detailed' if details else 'layer_names_only',
            'int8_layer_count':len(int8),'int8_layers':int8}
        if not details:raise RuntimeError('cached TRT engine inspector exposes layer names only; precision metadata unavailable')
        if not int8:raise RuntimeError('TRT engine has no reported INT8 compute layer')
        baseline=read(ROOT/'artifacts/metal_nut/fp32/validation_quality.json')
        candidate_quality=read(package/'validation_quality.json')
        report['validation_quality']=candidate_quality
        base=baseline['metrics'];q=candidate_quality['metrics']
        if candidate_quality['coverage']['ERROR'] or candidate_quality['status']!='passed':raise RuntimeError('incomplete validation predictions')
        if base['AUROC']-q['AUROC']>cfg['quality']['max_auroc_drop'] or q['FNR']-base['FNR']>cfg['quality']['max_fnr_increase']:
            raise RuntimeError('validation quality gate failed')
        if report['cpu_heavy_fallback']:raise RuntimeError('CPU heavy-operator fallback observed')
        report.update(status='passed',int8_execution_verified=True,reason='ORT TensorRT target execution, cached-engine INT8 layer, and validation quality all verified')
    except Exception as exc:report['reason']=str(exc)
    save(run_dir/'evidence.json',report)
    save(ROOT/pointer_output,{'run_dir':str(run_dir),'evidence_sha256':sha256(run_dir/'evidence.json'),
        'status':report['status'],'int8_execution_verified':report['int8_execution_verified'],'reason':report['reason']})
    print(report['reason'])
    return 0 if report['int8_execution_verified'] else 3

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/metal_nut.yaml');p.add_argument('--candidate',default='int8_minmax');p.add_argument('--output',default='reports/trt_int8_evidence.json');a=p.parse_args()
    raise SystemExit(run(ROOT/a.config,a.candidate,a.output))
