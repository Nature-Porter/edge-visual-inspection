"""Sequential, fail-closed stage runner. It never downloads dependencies or data."""
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from common import ROOT, config, read, save, sha256

PY_TRAIN=str(ROOT/'.venv-train/Scripts/python.exe')
PY_TOOLS=str(ROOT/'.venv-tools/Scripts/python.exe')
CFG='configs/metal_nut.yaml'
STAGES={
 'P0':[[PY_TOOLS,'tools/environment.py','--output','reports/environment-current.json'],
       ['powershell','-NoProfile','-File','scripts/build_windows.ps1'],
       ['powershell','-NoProfile','-File','scripts/probe_cuda.ps1']],
 'P1':[[PY_TOOLS,'tools/prepare_data.py','--config',CFG],[PY_TRAIN,'tools/train.py','--config',CFG]],
 'P2':[[PY_TRAIN,'tools/export_onnx.py','--config',CFG],[PY_TRAIN,'tools/verify_alignment.py','--config',CFG,'--split','validation','--cpp']],
 'P3':[[PY_TOOLS,'tools/evaluate.py','--config',CFG,'--split','validation'],
       ['powershell','-NoProfile','-File','scripts/run_windows.ps1','run','--manifest','artifacts/metal_nut/fp32/model_manifest.json','--source','data/replay.json','--backend','ort-cuda','--fps','20','--output','runs/baseline']],
 'P4':[[PY_TOOLS,'tools/quantize.py','--config',CFG,'--phase','base'],[PY_TOOLS,'tools/activation_diagnostics.py'],[PY_TOOLS,'tools/quantize.py','--config',CFG,'--phase','selective'],
       *[[PY_TOOLS,'tools/evaluate.py','--config',CFG,'--split','validation','--manifest',f'artifacts/metal_nut/{name}/model_manifest.json'] for name in ['fp16','int8_minmax','int8_percentile','int8_selective_1','int8_selective_2','int8_selective_3']]],
 'P5':[[PY_TOOLS,'tools/verify_trt_route.py']],
 'P6':[['powershell','-NoProfile','-File','scripts/run_windows.ps1','benchmark','--manifest','artifacts/metal_nut/fp32/model_manifest.json','--source','data/replay.json','--backend','ort-cuda','--mode','pipeline','--postprocess','cpu','--warmup','100','--iterations','1000','--output','runs/bench-cpu-post'],
       ['powershell','-NoProfile','-File','scripts/run_windows.ps1','benchmark','--manifest','artifacts/metal_nut/fp32/model_manifest.json','--source','data/replay.json','--backend','ort-cuda','--mode','pipeline','--postprocess','cuda','--warmup','100','--iterations','1000','--output','runs/bench-cuda-post']],
 'P7':[[PY_TOOLS,'tools/freeze_candidates.py','--config',CFG],
       ['powershell','-NoProfile','-File','scripts/run_windows.ps1','run','--manifest','artifacts/metal_nut/fp32/model_manifest.json','--source','data/replay.json','--backend','ort-cuda','--fps','20','--loop','--duration-seconds','1800','--output','runs/soak-highres']],
 'P8':[[PY_TOOLS,'tools/validate_run.py','--run','runs/soak-highres','--output','reports/soak-highres-validation.json'],
       [PY_TOOLS,'tools/summarize_memory.py'],[PY_TOOLS,'tools/summarize_final_quality.py'],
       [PY_TOOLS,'tools/report_failures.py'],[PY_TOOLS,'tools/summarize_postprocess.py']]}

INPUTS={
 'P0':['CMakeLists.txt','scripts/build_windows.ps1'],
 'P1':[CFG,'tools/prepare_data.py','tools/train.py','data/raw/imagenette2.tgz','data/train_normal.json','data/stats_normal.json'],
 'P2':[CFG,'tools/export_onnx.py','artifacts/metal_nut/training/model.ckpt','data/validation.json'],
 'P3':['artifacts/metal_nut/fp32/model_manifest.json','data/replay.json'],
 'P4':['artifacts/metal_nut/fp32/model.onnx','data/calibration.json','data/validation.json'],
 'P5':['artifacts/metal_nut/fp32/model.onnx','reports/quantization.json',
       'artifacts/metal_nut/int8_selective_2_trt_compat/model.onnx',
       'runs/trt-detailed-int8_selective_2_trt_compat/model.engine',
       'reports/trt_detailed_exploratory_selective_2.json',
       'reports/trt_direct_validation_quality_selective_2.json',
       'tools/verify_trt_route.py'],
 'P6':['artifacts/metal_nut/fp32/model.onnx','data/replay.json'],
 'P7':[CFG,'artifacts/metal_nut/fp32/validation_quality.json','reports/quantization.json','data/test.json','data/replay.json'],
 'P8':['docs/STATUS.md','docs/DECISIONS.md','docs/ACCEPTANCE.md','docs/REPORT.md','docs/DEMO.md']}

def fingerprints(stage):
    result={}
    for relative in INPUTS[stage]:
        path=ROOT/relative
        if not path.is_file():raise FileNotFoundError(f'{stage} requires {path}')
        result[relative]=sha256(path)
    return result

def run_stage(stage,record):
    try: inputs=fingerprints(stage)
    except FileNotFoundError as e:return {'status':'blocked','reason':str(e),'commands':[]}
    previous=record.get(stage)
    if stage!='P5' and previous and previous.get('status')=='passed' and previous.get('inputs')==inputs:
        return previous
    if stage=='P0':
        build_log=ROOT/'reports/build-release-fixed.log'
        cuda_log=ROOT/'reports/cuda-smoke.log'
        profiles=list((ROOT/'runs/p0-conv-cuda').glob('ort-profile_*.json'))
        executable=ROOT/'build/inspection.exe'
        environment=ROOT/'reports/environment-current.json'
        if all(p.is_file() for p in [build_log,cuda_log,executable,environment]) and profiles:
            has_cuda_conv=any(e.get('cat')=='Node' and e.get('args',{}).get('provider')=='CUDAExecutionProvider' and e['args'].get('op_name')=='Conv' for e in read(profiles[0]))
            if '100% tests passed, 0 tests failed out of 2' in build_log.read_text(encoding='utf-8',errors='replace') and 'CUDA add_one result=42' in cuda_log.read_text(encoding='utf-8',errors='replace') and has_cuda_conv:
                return {'status':'passed','inputs':inputs,'commands':[],'adopted_existing_p0_evidence':True,'outputs':{str(p.relative_to(ROOT)):sha256(p) for p in [build_log,cuda_log,profiles[0],executable,environment]}}
    if stage=='P1':
        checkpoint=ROOT/'artifacts/metal_nut/training/model.ckpt'
        report=ROOT/'artifacts/metal_nut/training/training.json'
        if checkpoint.is_file() and report.is_file():
            trained=read(report)
            if trained.get('status')=='trained' and trained.get('steps')==config(CFG)['training']['max_steps'] and trained.get('checkpoint_sha256')==sha256(checkpoint) and trained.get('train_sha256')==sha256(ROOT/'data/train_normal.json') and trained.get('stats_sha256')==sha256(ROOT/'data/stats_normal.json'):
                return {'status':'passed','inputs':inputs,'commands':[],'adopted_existing_training':True,'outputs':{'artifacts/metal_nut/training/model.ckpt':sha256(checkpoint),'artifacts/metal_nut/training/training.json':sha256(report)}}
    if stage=='P3':
        quality_path=ROOT/'artifacts/metal_nut/fp32/validation_quality.json'
        run=ROOT/'runs/baseline'
        audit_path=ROOT/'reports/baseline-validation.json'
        files=[quality_path,run/'snapshot.json',run/'summary.json',run/'events.jsonl',audit_path]
        if all(path.is_file() for path in files):
            quality=read(quality_path);snapshot=read(run/'snapshot.json')
            summary=read(run/'summary.json');audit=read(audit_path)
            model=read(ROOT/'artifacts/metal_nut/fp32/model_manifest.json')
            if (quality.get('status')=='passed' and quality.get('model_sha256')==model['onnx_sha256']
                and quality.get('data_sha256')==sha256(ROOT/'data/validation.json')
                and quality['coverage']['valid']==69 and quality['coverage']['ERROR']==0
                and snapshot.get('manifest_sha256')==sha256(ROOT/'artifacts/metal_nut/fp32/model_manifest.json')
                and snapshot.get('source')==read(ROOT/'data/replay.json')['records']
                and snapshot.get('args',{}).get('--backend')=='ort-cuda'
                and summary.get('generated')==69 and summary.get('completed')==69
                and summary.get('error')==0 and summary.get('queue_overflow')==0
                and audit.get('status')=='passed' and audit.get('event_count')==69
                and audit.get('events_sha256')==sha256(run/'events.jsonl')
                and audit.get('summary_sha256')==sha256(run/'summary.json')):
                return {'status':'passed','inputs':inputs,'commands':[],'adopted_existing_baseline':True,
                    'outputs':{path.relative_to(ROOT).as_posix():sha256(path) for path in files}}
    if stage=='P6':
        path=ROOT/'reports/postprocess_ab.json'
        if path.is_file():
            evidence=read(path)
            model=read(ROOT/'artifacts/metal_nut/fp32/model_manifest.json')
            compare=evidence.get('score_comparison',{})
            if evidence.get('model_sha256')==model['onnx_sha256'] and compare.get('n')==69 and compare.get('max_abs')==0 and compare.get('same_status_decision')==69:
                rounds=evidence.get('three_rounds',{})
                if all(len(rounds.get(post,{}).get('runs',[]))==3 and all((ROOT/item['summary']).is_file() and sha256(ROOT/item['summary'])==item['sha256'] for item in rounds[post]['runs']) for post in ('cpu','cuda')):
                    return {'status':'passed','inputs':inputs,'commands':[],'adopted_existing_ab_evidence':True,'outputs':{'reports/postprocess_ab.json':sha256(path)}}
    logdir=ROOT/'reports/pipeline';logdir.mkdir(parents=True,exist_ok=True)
    result={'status':'in_progress','started_utc':datetime.now(timezone.utc).isoformat(),'inputs':inputs,'commands':[]}
    log_stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    commands=list(STAGES[stage])
    if stage=='P7' and (ROOT/'artifacts/frozen.json').is_file():
        frozen=read(ROOT/'artifacts/frozen.json')
        if frozen.get('config_sha256')!=sha256(ROOT/CFG) or frozen.get('test_sha256')!=sha256(ROOT/'data/test.json') or frozen.get('validation_sha256')!=sha256(ROOT/'data/validation.json'):
            raise ValueError('existing freeze fingerprint mismatch')
        for manifest,digest in frozen['manifests'].items():
            if sha256(ROOT/manifest)!=digest:raise ValueError(f'frozen manifest changed: {manifest}')
        tests=[]
        for manifest in frozen['manifests']:
            path=ROOT/manifest;quality_path=path.parent/'test_quality.json'
            if quality_path.is_file():
                quality=read(quality_path)
                if quality.get('status')=='passed' and quality.get('model_sha256')==read(path)['onnx_sha256'] and quality.get('data_sha256')==frozen['test_sha256'] and quality['coverage']['ERROR']==0 and quality['coverage']['valid']==quality['coverage']['expected']:
                    continue
            tests.append([PY_TOOLS,'tools/evaluate.py','--config',CFG,'--split','test','--manifest',manifest])
        commands=tests+[commands[-1]]
    if stage=='P4':
        report_path=ROOT/'reports/quantization.json'
        required={'fp16','int8_minmax','int8_percentile'}
        if report_path.is_file() and required <= {r['candidate'] for r in read(report_path) if r['status']=='generated'}:
            base_manifest=read(ROOT/'artifacts/metal_nut/fp32/model_manifest.json')
            reusable=True
            for name in required:
                candidate=ROOT/'artifacts/metal_nut'/name
                if not (candidate/'model_manifest.json').is_file() or not (candidate/'model.onnx').is_file():
                    reusable=False;break
                m=read(candidate/'model_manifest.json')
                reusable &= m.get('checkpoint_sha256')==base_manifest['checkpoint_sha256'] and m.get('onnx_sha256')==sha256(candidate/'model.onnx')
            if reusable:commands=commands[1:]
    for index,command in enumerate(commands):
        path=logdir/f'{stage}-{log_stamp}-{index}.log'
        with path.open('w',encoding='utf-8',errors='replace') as log:
            log.write('COMMAND: '+' '.join(command)+'\n');log.flush()
            completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        item={'argv':command,'log':path.relative_to(ROOT).as_posix(),'exit_code':completed.returncode,'log_sha256':sha256(path)}
        result['commands'].append(item)
        if completed.returncode:
            result.update(status='blocked' if completed.returncode==3 else 'failed',reason=f'command {index} exited {completed.returncode}')
            break
        if stage=='P7' and command[1]=='tools/freeze_candidates.py':
            frozen=read(ROOT/'artifacts/frozen.json')
            tests=[[PY_TOOLS,'tools/evaluate.py','--config',CFG,'--split','test','--manifest',manifest] for manifest in frozen['manifests']]
            commands[1:1]=tests
    else:
        # Presence and content checks supplement process exit codes.
        outputs={
          'P0':['build/inspection.exe','reports/environment-current.json'],
          'P1':['artifacts/metal_nut/training/model.ckpt','artifacts/metal_nut/training/training.json'],
          'P2':['artifacts/metal_nut/fp32/model.onnx','artifacts/metal_nut/fp32/model_manifest.json','reports/alignment.json'],
          'P3':['artifacts/metal_nut/fp32/validation_quality.json','runs/baseline/summary.json'],
          'P4':['reports/quantization.json'],
          'P5':['reports/trt_probe.json','reports/trt_verified_selective_2.json'],
          'P6':['runs/bench-cpu-post/summary.json','runs/bench-cuda-post/summary.json'],
          'P7':['artifacts/metal_nut/fp32/test_quality.json','runs/soak-highres/summary.json'],
          'P8':['README.md','docs/ACCEPTANCE.md','docs/REPORT.md','docs/DEMO.md','reports/soak-highres-validation.json','reports/soak-highres-memory-summary.json','reports/final_quality.json','reports/failure_samples.json','reports/postprocess_ab.json']}
        missing=[p for p in outputs[stage] if not (ROOT/p).is_file()]
        if missing:result.update(status='failed',reason='missing outputs: '+','.join(missing))
        else:
            result.update(status='passed',outputs={p:sha256(ROOT/p) for p in outputs[stage]})
            if stage=='P4':
                names=['fp16','int8_minmax','int8_percentile','int8_selective_1','int8_selective_2','int8_selective_3']
                bad=[]
                for name in names:
                    quality=ROOT/'artifacts/metal_nut'/name/'validation_quality.json'
                    if not quality.is_file() or read(quality).get('status')!='passed' or read(quality)['coverage']['ERROR']:
                        bad.append(name)
                if bad:result.update(status='failed',reason='candidate validation missing or failed: '+','.join(bad))
            if stage=='P5':
                route=read(ROOT/'reports/trt_verified_selective_2.json')
                if (route.get('status')!='passed_validation_experiment'
                    or not route.get('int8_execution_verified')
                    or not route.get('validation_quality_pass')):
                    result.update(status='blocked',reason='direct TensorRT target INT8 execution or validation quality not verified')
            if stage=='P7':
                frozen=read(ROOT/'artifacts/frozen.json')
                bad=[]
                for manifest in frozen['manifests']:
                    quality=(ROOT/manifest).parent/'test_quality.json'
                    if not quality.is_file() or read(quality).get('status')!='passed' or read(quality)['coverage']['ERROR']:
                        bad.append(manifest)
                soak=read(ROOT/'runs/soak-highres/summary.json')
                if bad or soak.get('elapsed_seconds',0)<1800 or not soak.get('latency_pass'):
                    result.update(status='failed',reason='final test or soak acceptance failed: '+','.join(bad))
            if stage=='P8' and read(ROOT/'reports/soak-highres-validation.json').get('status')!='passed':
                result.update(status='failed',reason='long replay event/schema validation failed')
    result['finished_utc']=datetime.now(timezone.utc).isoformat()
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=CFG);p.add_argument('--stage',choices=[*STAGES,'all'],default='all');a=p.parse_args();config(a.config)
    file=ROOT/'reports/pipeline_states.json';record=read(file) if file.exists() else {}
    stages=list(STAGES) if a.stage=='all' else [a.stage]
    blocked=False
    for stage in stages:
        # P4 needs candidate validation; P5 requires the actual target engine and quality evidence.
        value=run_stage(stage,record)
        # A long-running stage may finish after another terminal recorded a shorter one.
        # Reload before updating this stage so unrelated results are not overwritten.
        record=read(file) if file.exists() else {}
        record[stage]=value;save(file,record);print(stage,value['status'],value.get('reason',''),flush=True)
        blocked |= value['status']!='passed'
    raise SystemExit(3 if blocked else 0)
