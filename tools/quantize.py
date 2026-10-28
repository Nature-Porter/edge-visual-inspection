import argparse
import copy
from collections import Counter
from common import ROOT,config,read,save,sha256,preprocess,verified_records

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/metal_nut.yaml');p.add_argument('--phase',choices=['base','selective','all'],default='all');a=p.parse_args();config(a.config)
    import onnx
    from onnxruntime.quantization import quantize_static,CalibrationDataReader,QuantFormat,QuantType,CalibrationMethod
    from onnxconverter_common import float16
    base=ROOT/'artifacts/metal_nut/fp32';manifest=read(base/'model_manifest.json');model=onnx.load(base/'model.onnx')
    if manifest.get('smoke'):raise ValueError('smoke model is not a production candidate')
    rows=verified_records('calibration');trained={r['source_image_id'] for r in verified_records('train_normal')}
    if not {r['source_image_id'] for r in rows}<=trained:raise ValueError('calibration outside train_normal')
    class Reader(CalibrationDataReader):
        def __init__(self):self.rewind()
        def rewind(self):self.items=iter(rows)
        def get_next(self):
            row=next(self.items,None)
            return None if row is None else {manifest['input']['name']:preprocess(ROOT/row['path'])}
    def package(name,precision,method=None,exclude=None):
        out=base.parent/name;out.mkdir(parents=True,exist_ok=True)
        if method is None:
            converted=float16.convert_float_to_float16(copy.deepcopy(model),keep_io_types=True);onnx.save(converted,out/'model.onnx')
        else:
            quantize_static(str(base/'model.onnx'),str(out/'model.onnx'),Reader(),quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,per_channel=True,reduce_range=False,
                calibrate_method=method,op_types_to_quantize=['Conv','MatMul'],nodes_to_exclude=exclude or [],
                extra_options={'ActivationSymmetric':True,'WeightSymmetric':True,'CalibTensorRangeSymmetric':True})
        graph=onnx.load(out/'model.onnx');onnx.checker.check_model(graph)
        m=copy.deepcopy(manifest);m.update(model_id='efficientad_s_'+name,onnx_sha256=sha256(out/'model.onnx'),requested_precision=precision,status='uncalibrated',image_threshold=None,pixel_threshold=None,validation_manifest_sha256=None)
        save(out/'model_manifest.json',m)
        save(out/'precision_intent.json',{'requested_precision':precision,'actual_execution_verified':False,'calibration_sha256':sha256(ROOT/'data/calibration.json'),
            'calibration_method':str(method),'excluded_nodes':exclude or [],'operators':dict(Counter(n.op_type for n in graph.graph.node)),
            'model_sha256':m['onnx_sha256'],'nodes':[{'name':n.name,'op':n.op_type,'intended':('FP32' if n.name in (exclude or []) else 'candidate_INT8') if method else 'FP16_with_FP32_boundaries'} for n in model.graph.node if n.op_type in ['Conv','MatMul']],
            'note':'QDQ and precision intention do not prove accelerated INT8 execution'})
    report_path=ROOT/'reports/quantization.json'
    if a.phase=='selective':
        results=read(report_path) if report_path.exists() else []
        required={'fp16','int8_minmax','int8_percentile'}
        if not required <= {r['candidate'] for r in results if r['status']=='generated'}:
            raise ValueError('base candidates must be generated before selective quantization')
        for name in required:
            candidate=base.parent/name
            previous=read(candidate/'model_manifest.json')
            if previous['checkpoint_sha256']!=manifest['checkpoint_sha256'] or sha256(candidate/'model.onnx')!=previous['onnx_sha256']:
                raise ValueError(f'stale base candidate: {name}')
        results=[r for r in results if r['candidate'] in required]
    else:
        results=[]
        for name,precision,method in [('fp16','FP16',None),('int8_minmax','INT8/FP32',CalibrationMethod.MinMax),('int8_percentile','INT8/FP32',CalibrationMethod.Percentile)]:
            try:package(name,precision,method);results.append({'candidate':name,'status':'generated'})
            except Exception as e:results.append({'candidate':name,'status':'failed','reason':str(e)})
    # Exclusions must be justified by activation diagnostics, not test quality.
    diagnostic=ROOT/'reports/activation_diagnostics.json'
    if a.phase!='base' and diagnostic.exists():
        diag=read(diagnostic)
        if diag['float_model_sha256']!=sha256(base/'model.onnx'):raise ValueError('stale activation diagnostic')
        nodes=diag['ranked_conv_nodes']
        for k in [1,2,3]:
            excluded=nodes[:k];name=f'int8_selective_{k}'
            try:package(name,'INT8/FP32',CalibrationMethod.MinMax,excluded);results.append({'candidate':name,'status':'generated'})
            except Exception as e:results.append({'candidate':name,'status':'failed','reason':str(e)})
    elif a.phase!='base':results.append({'candidate':'selective','status':'blocked','reason':'run activation_diagnostics.py on generated MinMax candidate first'})
    save(report_path,results)
    print(results)
    return 0 if all(r['status']=='generated' for r in results) else 3
if __name__=='__main__':raise SystemExit(main())
