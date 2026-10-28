import argparse
from pathlib import Path
from common import ROOT,read,save,config,sha256,verified_records,preprocess,ort_session
from metrics import metrics,calibrate,join_predictions

def evaluate(c,manifest_path,split):
    import numpy as np
    m=read(manifest_path);path=manifest_path.parent/m['onnx_path']
    if sha256(path)!=m['onnx_sha256']:raise ValueError('model hash changed')
    if split=='test':
        frozen=read(ROOT/'artifacts/frozen.json')
        key=manifest_path.relative_to(ROOT).as_posix()
        if frozen['manifests'].get(key)!=sha256(manifest_path) or frozen['test_sha256']!=sha256(ROOT/'data/test.json') or frozen['config_sha256']!=sha256(ROOT/'configs/metal_nut.yaml'):raise ValueError('final test requires unchanged frozen candidates, config and test manifest')
        if m.get('smoke'):raise ValueError('smoke is not a final quality model')
    rows=verified_records(split);session=ort_session(path);predictions=[]
    for row in rows:
        try:
            y=session.run([m['output']['name']],{m['input']['name']:preprocess(ROOT/row['path'])})[0]
            if list(y.shape)!=m['output']['shape'] or not np.isfinite(y).all():raise ValueError('INVALID_OUTPUT')
            predictions.append({'source_image_id':row['source_image_id'],'score':float(y.max()),'status':'SUCCESS'})
        except Exception as e:predictions.append({'source_image_id':row['source_image_id'],'score':None,'status':'ERROR','reason':str(e)})
    valid,coverage=join_predictions(rows,predictions)
    report={'split':split,'coverage':coverage,'model_sha256':m['onnx_sha256'],'data_sha256':sha256(ROOT/'data'/f'{split}.json'),'pixel_metrics':None,'pixel_reason':'pixel threshold uncalibrated'}
    if coverage['ERROR']:
        report.update(status='failed',metrics=None)
    else:
        labels,scores=zip(*valid)
        if split=='validation':
            threshold,result=calibrate(labels,scores,c['quality']['validation_fpr_budget'])
            m.update(image_threshold=threshold,status='calibrated',validation_manifest_sha256=report['data_sha256'])
            save(manifest_path,m)
        else:
            if m['status']!='calibrated':raise ValueError('uncalibrated final candidate')
            result=metrics(labels,scores,m['image_threshold'])
        report.update(status='passed',metrics=result)
    save(manifest_path.parent/f'{split}_predictions.json',predictions);save(manifest_path.parent/f'{split}_quality.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/metal_nut.yaml');p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--manifest',default='artifacts/metal_nut/fp32/model_manifest.json');a=p.parse_args()
    report=evaluate(config(a.config),ROOT/a.manifest,a.split);print(report);raise SystemExit(0 if report['status']=='passed' else 5)
