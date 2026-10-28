"""Report frozen final-test outcomes without changing candidate selection."""
import math
from common import ROOT,config,read,save,sha256

def wilson95(hits,total):
    if total<=0:return None
    z=1.959963984540054;p=hits/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-radius,center+radius]

def main():
    cfg=config(ROOT/'configs/metal_nut.yaml')
    frozen_path=ROOT/'artifacts/frozen.json';frozen=read(frozen_path)
    if frozen['test_sha256']!=sha256(ROOT/'data/test.json'):
        raise ValueError('final test manifest changed after freeze')
    base=read(ROOT/'artifacts/metal_nut/fp32/test_quality.json')
    if base['status']!='passed' or base['coverage']['ERROR']:
        raise ValueError('FP32 final quality incomplete')
    reference=base['metrics'];rows=[]
    for relative,digest in frozen['manifests'].items():
        manifest_path=ROOT/relative
        if sha256(manifest_path)!=digest:raise ValueError(f'frozen manifest changed: {relative}')
        manifest=read(manifest_path)
        quality=read(manifest_path.parent/'test_quality.json')
        if quality['status']!='passed' or quality['coverage']['ERROR'] or quality['coverage']['expected']!=quality['coverage']['valid']:
            raise ValueError(f'final quality incomplete: {relative}')
        actual=quality['metrics'];auroc_drop=reference['AUROC']-actual['AUROC'];fnr_increase=actual['FNR']-reference['FNR']
        rows.append({'model_id':manifest['model_id'],'model_sha256':manifest['onnx_sha256'],'manifest':relative,
            'metrics':actual,'wilson95_fnr':wilson95(actual['FN'],actual['TP']+actual['FN']),
            'wilson95_fpr':wilson95(actual['FP'],actual['TN']+actual['FP']),
            'relative_auroc_drop':auroc_drop,'relative_fnr_increase':fnr_increase,
            'final_quality_gate':auroc_drop<=cfg['quality']['max_auroc_drop'] and fnr_increase<=cfg['quality']['max_fnr_increase'] and actual['FPR']<=cfg['quality']['validation_fpr_budget'],
            'int8_execution_verified':False})
    save(ROOT/'reports/final_quality.json',{'frozen_sha256':sha256(frozen_path),'test_sha256':frozen['test_sha256'],
        'reference':'efficientad_s_fp32','unique_test_images':reference['n'],'candidates':rows,
        'note':'All candidates were frozen before reading test; this report does not revise selection or thresholds.'})
    print('reports/final_quality.json')

if __name__=='__main__':main()
