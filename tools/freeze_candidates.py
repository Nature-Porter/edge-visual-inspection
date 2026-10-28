import argparse
from common import ROOT,config,read,save,sha256
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/metal_nut.yaml');a=p.parse_args();c=config(a.config)
    path=ROOT/'artifacts/frozen.json'
    if path.exists():raise FileExistsError('freeze already exists: final test protocol cannot silently restart')
    base=ROOT/'artifacts/metal_nut';reference=read(base/'fp32/validation_quality.json');ref=reference['metrics']
    if reference['status']!='passed' or reference['coverage']['ERROR']:raise ValueError('FP32 validation incomplete')
    selection=[];manifests={}
    for file in sorted(base.glob('*/model_manifest.json')):
        m=read(file)
        if m.get('smoke') or m.get('fixture'):continue
        qpath=file.parent/'validation_quality.json'
        if not qpath.exists():continue
        q=read(qpath);valid=q['status']=='passed' and q['coverage']['ERROR']==0
        if valid:
            met=q['metrics'];valid=ref['AUROC']-met['AUROC']<=c['quality']['max_auroc_drop'] and met['FNR']-ref['FNR']<=c['quality']['max_fnr_increase'] and met['FPR']<=c['quality']['validation_fpr_budget']
        selection.append({'model_id':m['model_id'],'quality_gate':valid,'validation':q,'bytes':(file.parent/m['onnx_path']).stat().st_size,'precision_execution_verified':False})
        if valid:
            assert sha256(file.parent/m['onnx_path'])==m['onnx_sha256']
            manifests[file.relative_to(ROOT).as_posix()]=sha256(file)
    if not manifests:raise ValueError('no eligible final candidates')
    save(path,{'schema_version':1,'config_sha256':sha256(a.config),'test_sha256':sha256(ROOT/'data/test.json'),
      'validation_sha256':sha256(ROOT/'data/validation.json'),'manifests':manifests,'selection':selection,
      'rule':'AUROC drop<=0.01 and FNR increase<=0.02; fixed per-candidate validation thresholds; no test-based selection'})
    print(path)
