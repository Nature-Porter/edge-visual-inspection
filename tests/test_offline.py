import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from metrics import calibrate,metrics,join_predictions
from common import ROOT,read,sha256

def test_threshold_inclusive_and_budget():
    threshold,result=calibrate([0,0,1,1],[.1,.4,.4,.9],0)
    assert threshold==pytest.approx(.9)
    assert result['FP']==0 and result['FN']==1
    assert metrics([0,1],[.5,.5],.5)['FP']==1

def test_missing_and_duplicate_quality_predictions():
    records=[{'source_image_id':'a','label':0},{'source_image_id':'b','label':1}]
    values,coverage=join_predictions(records,[{'source_image_id':'a','score':.1}])
    assert coverage['ERROR']==1 and coverage['coverage']==.5
    with pytest.raises(ValueError):join_predictions(records,[{'source_image_id':'a','score':.1}]*2)
    with pytest.raises(ValueError):calibrate([0,1],[np.nan,.2],.1)

def test_actual_split_disjointness_and_calibration():
    splits={k:read(ROOT/'data'/f'{k}.json')['records'] for k in ['train_normal','stats_normal','dev_normal','calibration','validation','test']}
    ids={k:{r['source_image_id'] for r in v} for k,v in splits.items()}
    assert ids['calibration']<=ids['train_normal']
    assert ids['dev_normal']<=ids['validation']
    assert not ids['validation']&ids['test']
    for a,b in [('train_normal','stats_normal'),('train_normal','validation'),('stats_normal','validation'),('train_normal','test'),('stats_normal','test')]:assert not ids[a]&ids[b]
    for k in ['validation','test']:assert {r['label'] for r in splits[k]}=={0,1}
    report=read(ROOT/'reports/data_protocol.json')
    for k in splits:assert sha256(ROOT/'data'/f'{k}.json')==report['splits'][k]['sha256']

def test_manifest_schema_rejects_uncalibrated_ok_claim():
    import jsonschema
    m=read(ROOT/'tests/fixtures/conv/model_manifest.json');schema=read(ROOT/'schemas/model_manifest.schema.json')
    jsonschema.validate(m,schema)
    m['status']='calibrated'
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(m,schema)
