"""Image-level metrics. Defect is positive. No repeated replay inflation."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def metrics(labels,scores,threshold):
    labels=np.asarray(labels,dtype=int);scores=np.asarray(scores,dtype=float)
    if len(labels)!=len(scores) or not np.isfinite(scores).all():raise ValueError('invalid scores')
    pred=scores>=threshold;pos=labels==1;neg=labels==0
    tp=int((pred&pos).sum());fn=int((~pred&pos).sum());fp=int((pred&neg).sum());tn=int((~pred&neg).sum())
    both=pos.any() and neg.any()
    return {'n':len(labels),'TP':tp,'FP':fp,'TN':tn,'FN':fn,'FNR':fn/int(pos.sum()) if pos.any() else None,
        'FPR':fp/int(neg.sum()) if neg.any() else None,'AUROC':float(roc_auc_score(labels,scores)) if both else None,
        'AP':float(average_precision_score(labels,scores)) if both else None,'AP_definition':'sklearn non-interpolated average precision',
        'undefined_reason':None if both else 'both classes required','image_threshold':float(threshold)}

def calibrate(labels,scores,budget):
    labels=np.asarray(labels);scores=np.asarray(scores,dtype=np.float32)
    if not (labels==0).any() or not (labels==1).any() or not np.isfinite(scores).all():raise ValueError('calibration requires finite scores and both classes')
    candidates=np.append(np.unique(scores),np.nextafter(scores.max(),np.float32(np.inf)))
    choices=[]
    for t in candidates:
        m=metrics(labels,scores,t)
        if m['FPR']<=budget:choices.append(((m['FNR'],m['FPR'],-float(t)),float(t),m))
    if not choices:raise ValueError('no feasible threshold')
    _,threshold,result=min(choices,key=lambda x:x[0]);return threshold,result

def join_predictions(records,predictions):
    expected={r['source_image_id']:r for r in records}
    if len(expected)!=len(records):raise ValueError('duplicate quality source IDs')
    seen={}
    for p in predictions:
        key=p['source_image_id']
        if key not in expected:raise ValueError('prediction outside quality manifest')
        if key in seen:raise ValueError('replay duplicates are not quality observations')
        seen[key]=p
    valid=[];errors=[]
    for key,r in expected.items():
        p=seen.get(key)
        if not p or p.get('score') is None or not np.isfinite(p['score']):errors.append(key)
        else:valid.append((r['label'],p['score']))
    return valid,{'expected':len(expected),'received':len(seen),'valid':len(valid),'ERROR':len(errors),'coverage':len(valid)/len(expected) if expected else 0,'error_ids':errors}
