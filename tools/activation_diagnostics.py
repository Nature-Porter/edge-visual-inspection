"""ORT public QDQ debug API, normal training calibration only, no final test."""
from common import ROOT,sha256,save,verified_records,preprocess

if __name__=='__main__':
    import onnx,numpy as np
    from onnxruntime.quantization import CalibrationDataReader
    from onnxruntime.quantization.qdq_loss_debug import modify_model_output_intermediate_tensors,collect_activations,create_activation_matching
    base=ROOT/'artifacts/metal_nut';fp=base/'fp32/model.onnx';qdq=base/'int8_minmax/model.onnx';out=ROOT/'artifacts/diagnostics';out.mkdir(parents=True,exist_ok=True)
    rows=verified_records('calibration')[:8]
    class Reader(CalibrationDataReader):
        def __init__(self):self.items=iter(rows)
        def get_next(self):
            row=next(self.items,None)
            return {'image':preprocess(ROOT/row['path'])} if row else None
    # Capture Q/DQ boundaries as well as Conv/MatMul outputs; without those
    # boundary tensors ORT's matcher has no pre/post pair to compare.
    for source,name in [(fp,'float'),(qdq,'qdq')]:modify_model_output_intermediate_tensors(str(source),str(out/(name+'.onnx')),op_types_for_saving=['Conv','MatMul','QuantizeLinear','DequantizeLinear'])
    float_acts=collect_activations(str(out/'float.onnx'),Reader(),execution_providers=['CPUExecutionProvider'])
    qdq_acts=collect_activations(str(out/'qdq.onnx'),Reader(),execution_providers=['CPUExecutionProvider'])
    matched=create_activation_matching(qdq_acts,float_acts);errors={}
    for name,values in matched.items():
        if 'float' in values and 'post_qdq' in values:
            a=np.concatenate([v.ravel() for v in values['float']]);b=np.concatenate([v.ravel() for v in values['post_qdq']])
            errors[name]=float(np.mean(np.abs(a-b))/(np.mean(np.abs(a))+1e-12))
    graph=onnx.load(fp);ranked=sorted([(max([errors.get(o,0) for o in n.output]),n.name) for n in graph.graph.node if n.op_type in ['Conv','MatMul']],reverse=True)
    if not errors:raise RuntimeError('no matched activations; cannot justify selective quantization')
    save(ROOT/'reports/activation_diagnostics.json',{'float_model_sha256':sha256(fp),'qdq_model_sha256':sha256(qdq),'calibration_sha256':sha256(ROOT/'data/calibration.json'),
        'diagnostic_provider':'CPUExecutionProvider (numerical diagnostic only)','relative_mae':errors,'ranked_conv_nodes':[n for _,n in ranked]})
