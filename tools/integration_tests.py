"""Real C++ process tests using explicitly synthetic ONNX inputs."""
import copy
import json
import os
import subprocess
import time
from pathlib import Path
from common import ROOT,read,save,sha256

def runtime_env():
    env=os.environ.copy()
    paths=[ROOT/'third_party/opencv/build/x64/vc16/bin',ROOT/'third_party/TensorRT-10.9.0.34/lib',Path(env.get('CUDA_PATH','C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5'))/'bin']
    paths+=list(Path('C:/Users/yuxiangw/ort-env/Lib/site-packages/nvidia').glob('*/bin'))
    env['PATH']=';'.join(map(str,paths))+';'+env['PATH']
    return env

def main():
    import numpy as np
    import onnx
    import jsonschema
    from make_fixture import make
    make();fixture=ROOT/'tests/fixtures/conv';base=read(fixture/'model_manifest.json');source=read(fixture/'replay.json')
    root=ROOT/'runs'/f'integration-{time.time_ns()}';root.mkdir(parents=True)
    reports=[]
    def case(name,manifest=None,records=None,extra=(),expected=0,check=None):
        folder=root/name;folder.mkdir()
        m=copy.deepcopy(base if manifest is None else manifest);m['onnx_path']=str((fixture/m['onnx_path']).resolve())
        save(folder/'manifest.json',m);save(folder/'source.json',source if records is None else {'records':records})
        cmd=[str(ROOT/'build/inspection.exe'),'run','--manifest',str(folder/'manifest.json'),'--source',str(folder/'source.json'),
            '--backend','ort-cuda','--postprocess','cuda','--fps','100','--deadline-ms','1000','--warmup','5','--output',str(folder/'output'),*extra]
        proc=subprocess.run(cmd,cwd=ROOT,env=runtime_env(),capture_output=True,text=True,errors='replace',timeout=120)
        (folder/'process.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
        ok=proc.returncode==expected;reason=''
        try:
            assert ok,f'exit {proc.returncode}, expected {expected}: {proc.stderr}'
            if expected==0:
                events=[json.loads(line) for line in (folder/'output/events.jsonl').read_text().splitlines()]
                summary=read(folder/'output/summary.json')
                assert len(events)==summary['generated']==summary['completed']
                assert len({e['frame_id'] for e in events})==len(events)
                assert {e['frame_id'] for e in events}==set(range(1,len(events)+1))
                for e in events:
                    jsonschema.validate(e,read(ROOT/'schemas/event.schema.json'))
                    assert e['status']=='SUCCESS' or e['decision']=='REVIEW'
                if check:check(events,summary)
        except Exception as e:ok=False;reason=str(e)
        reports.append({'name':name,'passed':ok,'command':cmd,'exit_code':proc.returncode,'reason':reason,'log':str(folder/'process.log')})
        print(name,ok,reason,flush=True)
        return folder
    def all_status(status):
        def check(events,summary):assert all(e['status']==status and e['decision']=='REVIEW' for e in events)
        return check
    warm=case('uncalibrated',extra=['--profile'],check=all_status('UNCALIBRATED'))
    profile=next((warm/'output').glob('ort-profile*.json'));nodes=[e['args'] for e in read(profile) if e.get('cat')=='Node' and e.get('args',{}).get('provider')]
    assert any(n['provider']=='CUDAExecutionProvider' and n.get('op_name')=='Conv' for n in nodes),'no CUDA Conv evidence'
    calibrated=copy.deepcopy(base);calibrated.update(status='calibrated',image_threshold=2.0,validation_manifest_sha256=base['training_manifest_sha256'])
    def ok_decisions(events,summary):assert all(e['decision']=='OK' and e['score'] is not None for e in events)
    case('calibrated_fixture',manifest=calibrated,check=ok_decisions)
    def overloaded(events,summary):assert summary['queue_overflow']>0 and all(e['decision']=='REVIEW' for e in events if e['status']=='QUEUE_OVERFLOW')
    case('overload',extra=['--fps','1000','--queue-capacity','1','--inject-delay-ms','30'],check=overloaded)
    def timeouts(events,summary):assert summary['timeout']>0 and all(e['decision']=='REVIEW' for e in events if e['status']=='TIMEOUT')
    case('timeouts',extra=['--deadline-ms','1','--inject-delay-ms','10'],check=timeouts)
    case('inference_errors',records=source['records'][:5],extra=['--fps','1','--inject-inference-error-every','1'],check=all_status('INFERENCE_ERROR'))
    bad=[{'path':str(fixture/'missing.png'),'source_image_id':'missing'}]
    case('bad_image',records=bad,check=all_status('INVALID_IMAGE'))
    m=copy.deepcopy(base);m['onnx_sha256']='f'*64;case('wrong_hash',manifest=m,expected=2)
    m=copy.deepcopy(base);m['output']['shape']=[1,1,10,10];case('wrong_shape',manifest=m,expected=2)
    case('write_failure',extra=['--inject-write-error-after','2'],expected=4)
    graph=onnx.load(fixture/'model.onnx');graph.graph.initializer[0].CopyFrom(onnx.numpy_helper.from_array(np.full((1,3,1,1),np.nan,np.float32),'weight'));onnx.save(graph,fixture/'nan.onnx')
    m=copy.deepcopy(base);m.update(onnx_path='nan.onnx',onnx_sha256=sha256(fixture/'nan.onnx'))
    case('nan_output',manifest=m,records=source['records'][:3],extra=['--fps','1'],check=all_status('INVALID_OUTPUT'))
    save(ROOT/'reports/integration.json',{'synthetic_only':True,'passed':all(r['passed'] for r in reports),'cases':reports,
        'cuda_evidence':{'profile':str(profile),'sha256':sha256(profile),'model_sha256':base['onnx_sha256'],'nodes':nodes}})
    return 0 if all(r['passed'] for r in reports) else 5
if __name__=='__main__':raise SystemExit(main())
