"""Fail closed until real TRT provider/engine precision evidence is available."""
import argparse
import os
import time
from pathlib import Path
from common import ROOT,read,save,sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/metal_nut.yaml');a=p.parse_args()
    import onnxruntime as ort
    from integration_tests import runtime_env
    os.environ.update(runtime_env())
    trt=ROOT/'third_party/TensorRT-10.9.0.34'
    for directory in (trt/'lib',trt/'bin'):
        if directory.is_dir():os.environ['PATH']=str(directory)+';'+os.environ['PATH']
    handles=[]
    if hasattr(os,'add_dll_directory'):
        for path in os.environ['PATH'].split(';'):
            if Path(path).is_dir():
                try:handles.append(os.add_dll_directory(path))
                except OSError:pass
    output=ROOT/'reports/trt_probe.json'
    dlls=[str(p) for folder in os.environ['PATH'].split(';') if Path(folder).is_dir() for p in Path(folder).glob('nvinfer*.dll')]
    report={'status':'blocked','ort_version':ort.__version__,'requested_combination':'ORT 1.22 / TRT 10.9 / CUDA12','dlls':dlls,
      'official_source':'https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html',
      'int8_execution_verified':False,'reason':None,'target_directory':str(trt)}
    if not dlls:
        report['reason']='TensorRT nvinfer DLLs not found in runtime search paths; CUDA EP availability does not imply TensorRT availability.'
        save(output,report);print(report['reason']);return 3
    try:
        import numpy as np
        from make_fixture import make
        make();model=ROOT/'tests/fixtures/conv/model.onnx';out=ROOT/'runs'/f'trt-probe-{time.time_ns()}';out.mkdir(parents=True)
        options=ort.SessionOptions();options.enable_profiling=True;options.profile_file_prefix=str(out/'profile')
        options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
        providers=[('TensorrtExecutionProvider',{'trt_dump_subgraphs':True,'trt_engine_cache_enable':False}),('CUDAExecutionProvider',{'use_tf32':False})]
        t=time.perf_counter();session=ort.InferenceSession(str(model),options,providers=providers);session.disable_fallback()
        report['cold_session_seconds']=time.perf_counter()-t
        session.run(None,{'image':np.ones((1,3,256,256),np.float32)});profile=session.end_profiling();nodes=[e.get('args',{}) for e in read(profile) if e.get('cat')=='Node']
        report.update(profile=profile,profile_sha256=sha256(profile),model_sha256=sha256(model),nodes=nodes)
        if not any(n.get('provider')=='TensorrtExecutionProvider' for n in nodes):raise RuntimeError('Conv not executed by TensorRT provider')
        report['conv_provider_verified']=True
        report['reason']='Conv TRT provider verified; target QDQ engine layer precision and validation quality still required.'
    except Exception as e:report['reason']=str(e)
    save(output,report);print(report['reason']);return 3
if __name__=='__main__':raise SystemExit(main())
