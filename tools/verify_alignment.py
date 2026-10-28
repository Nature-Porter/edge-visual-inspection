import argparse
import subprocess
from common import ROOT,config,preprocess,verified_records,ort_session,read,save,sha256

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/metal_nut.yaml');p.add_argument('--split',choices=['validation'],default='validation');p.add_argument('--cpp',action='store_true');a=p.parse_args();c=config(a.config)
    import torch,numpy as np
    from model import restore,Core
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=Core(restore(c,ROOT/'artifacts/metal_nut/training/model.ckpt')).cuda().eval();package=ROOT/'artifacts/metal_nut/fp32';m=read(package/'model_manifest.json')
    session=ort_session(package/m['onnx_path'],ROOT/'reports/alignment_profile');rows=verified_records('validation');result=[];all_pass=True
    fixture=ROOT/'tests/fixtures/alignment';fixture.mkdir(parents=True,exist_ok=True)
    for row in rows:
        x=preprocess(ROOT/row['path'])
        with torch.no_grad():y=model(torch.from_numpy(x).cuda()).cpu().numpy()
        z=session.run(None,{'image':x})[0];ok=y.shape==z.shape and np.isfinite(z).all() and np.allclose(y,z,atol=c['alignment']['atol'],rtol=c['alignment']['rtol'])
        all_pass &= bool(ok)
        result.append({'source_image_id':row['source_image_id'],'pass':bool(ok),'mae':float(np.abs(y-z).mean()),'max_abs':float(np.abs(y-z).max()),'score_difference':float(abs(y.max()-z.max()))})
        if len(result)==1:x.tofile(fixture/'input.f32');z.tofile(fixture/'raw_map.f32')
    report={'status':'passed' if all_pass else 'failed','same_tensor':result,'atol':c['alignment']['atol'],'rtol':c['alignment']['rtol'],'profile':session.end_profiling(),'model_sha256':m['onnx_sha256'],'cpp_full_chain':'not_run'}
    if a.cpp:
        import json,time
        output=ROOT/'runs'/f'alignment-{time.time_ns()}'
        cmd=['powershell','-NoProfile','-File',str(ROOT/'scripts/run_windows.ps1'),'run','--manifest',str(package/'model_manifest.json'),'--source','data/replay.json','--backend','ort-cuda','--fps','1','--deadline-ms','10000','--output',str(output)]
        subprocess.run(cmd,cwd=ROOT,check=True)
        events=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
        scores={e['source_image_id']:e['score'] for e in events}
        if len(scores)!=len(rows) or any(v is None for v in scores.values()):raise ValueError('C++ missing predictions')
        cpp=[]
        for row in rows:
            score=float(session.run(None,{'image':preprocess(ROOT/row['path'])})[0].max());difference=abs(scores[row['source_image_id']]-score)
            ok=bool(np.isclose(scores[row['source_image_id']],score,atol=c['alignment']['atol'],rtol=c['alignment']['rtol']));all_pass &= ok
            cpp.append({'source_image_id':row['source_image_id'],'score_difference':difference,'pass':ok})
        report['cpp_full_chain']=cpp;report['status']='passed' if all_pass else 'failed'
    save(ROOT/'reports/alignment.json',report);raise SystemExit(0 if all_pass else 5)
