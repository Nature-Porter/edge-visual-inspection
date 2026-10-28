"""List frozen-test errors for inspection, without revising thresholds or selection."""
from common import ROOT,read,save,sha256

def main():
    frozen=read(ROOT/'artifacts/frozen.json')
    if frozen['test_sha256']!=sha256(ROOT/'data/test.json'):raise ValueError('test manifest changed')
    records={r['source_image_id']:r for r in read(ROOT/'data/test.json')['records']}
    output=[]
    for relative,digest in frozen['manifests'].items():
        path=ROOT/relative
        if sha256(path)!=digest:raise ValueError(f'changed manifest: {relative}')
        manifest=read(path);threshold=manifest['image_threshold']
        predictions=read(path.parent/'test_predictions.json')
        if len(predictions)!=len(records):raise ValueError('incomplete predictions')
        false_negatives=[];false_positives=[]
        for prediction in predictions:
            item=records[prediction['source_image_id']]
            if prediction['status']!='SUCCESS' or prediction['score'] is None:raise ValueError('failed prediction is not a quality sample')
            predicted=int(prediction['score']>=threshold)
            if predicted==item['label']:continue
            case={'source_image_id':item['source_image_id'],'path':item['path'],'defect_type':item['defect_type'],
                  'score':prediction['score'],'threshold':threshold,'label':item['label']}
            (false_negatives if item['label'] else false_positives).append(case)
        output.append({'model_id':manifest['model_id'],'model_sha256':manifest['onnx_sha256'],
                       'false_negatives':false_negatives,'false_positives':false_positives})
    save(ROOT/'reports/failure_samples.json',{'frozen_sha256':sha256(ROOT/'artifacts/frozen.json'),
        'test_sha256':frozen['test_sha256'],'candidates':output,
        'note':'Post-hoc analysis only; no model or threshold is changed from the frozen protocol.'})
    print('reports/failure_samples.json')

if __name__=='__main__':main()
