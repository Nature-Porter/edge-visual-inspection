"""Same-backend C++ model benchmark for frozen precision candidates."""
import argparse
import subprocess
from statistics import median
from common import ROOT,read,save,sha256
from integration_tests import runtime_env

NAMES=['fp32','fp16','int8_percentile','int8_selective_2','int8_selective_3']

def main():
    p=argparse.ArgumentParser();p.add_argument('--warmup',type=int,default=100);p.add_argument('--iterations',type=int,default=1000);a=p.parse_args()
    if a.warmup<0 or a.iterations<=0:raise ValueError('invalid benchmark counts')
    base=ROOT/'artifacts/metal_nut';result={name:{'runs':[]} for name in NAMES}
    report_path=ROOT/'reports/precision_benchmarks.json'
    previous=read(report_path) if report_path.is_file() else None
    for round_number in range(1,4):
        order=NAMES[round_number-1:]+NAMES[:round_number-1]
        for name in order:
            manifest_path=base/name/'model_manifest.json';manifest=read(manifest_path)
            folder=ROOT/'runs'/f'precision-{name}-{round_number}';summary_path=folder/'summary.json'
            command=[str(ROOT/'build/inspection.exe'),'benchmark','--manifest',str(manifest_path),'--source','data/replay.json',
                '--backend','ort-cuda','--mode','model','--postprocess','cuda','--warmup',str(a.warmup),'--iterations',str(a.iterations),'--output',str(folder)]
            log_path=ROOT/'reports'/f'precision-{name}-{round_number}.log'
            if summary_path.is_file():
                summary=read(summary_path)
                if summary['model_sha256']!=manifest['onnx_sha256'] or summary['wall_ms']['count']!=a.iterations:raise ValueError(f'stale benchmark: {summary_path}')
                status='passed'
            elif folder.exists():
                old_runs=(previous or {}).get('candidates',{}).get(name,{}).get('runs',[])
                old=next((item for item in old_runs if item['round']==round_number),None)
                if (old and old['status']=='failed' and log_path.is_file() and old.get('log_sha256')==sha256(log_path)
                    and previous['candidates'][name]['model_sha256']==manifest['onnx_sha256']
                    and previous.get('warmup')==a.warmup and previous.get('iterations_per_round')==a.iterations
                    and old.get('command')==command):
                    status='failed'
                else:
                    raise FileExistsError(f'incomplete or stale benchmark directory: {folder}')
            else:
                completed=subprocess.run(command,cwd=ROOT,env=runtime_env(),capture_output=True,text=True,errors='replace',timeout=300)
                log_path.write_text('COMMAND: '+' '.join(command)+'\n'+completed.stdout+'\n'+completed.stderr,encoding='utf-8')
                status='passed' if completed.returncode==0 and summary_path.is_file() else 'failed'
                if status=='passed':summary=read(summary_path)
            record={'round':round_number,'status':status,'command':command,'log':str(log_path.relative_to(ROOT)),
                'log_sha256':sha256(log_path) if log_path.is_file() else None,
                'summary':str(summary_path.relative_to(ROOT)) if status=='passed' else None}
            if status=='passed':
                record.update(wall_p50_ms=summary['wall_ms']['p50'],wall_p95_ms=summary['wall_ms']['p95'],
                    model_gpu_p50_ms=summary['model_gpu_ms']['p50'])
            result[name]['runs'].append(record)
            print(name,round_number,status,flush=True)
    for name in NAMES:
        runs=[r for r in result[name]['runs'] if r['status']=='passed']
        result[name]['model_sha256']=read(base/name/'model_manifest.json')['onnx_sha256']
        result[name]['median_wall_p50_ms']=median(r['wall_p50_ms'] for r in runs) if len(runs)==3 else None
        result[name]['median_model_gpu_p50_ms']=median(r['model_gpu_p50_ms'] for r in runs) if len(runs)==3 else None
    save(report_path,{'backend':'C++ ORT CUDA','mode':'model','postprocess':'cuda','warmup':a.warmup,
        'iterations_per_round':a.iterations,'rounds':3,'candidates':result,
        'precision_note':'ORT CUDA execution time does not prove INT8 arithmetic; TensorRT layer evidence is separate.'})
    print('reports/precision_benchmarks.json')
    raise SystemExit(0 if all(len([r for r in result[name]['runs'] if r['status']=='passed'])==3 for name in NAMES) else 5)

if __name__=='__main__':main()
