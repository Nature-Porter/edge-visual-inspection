import argparse
import importlib.metadata
from collections import Counter
from common import ROOT,config,save,sha256,preprocess,verified_records,ort_session

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/metal_nut.yaml');p.add_argument('--smoke',action='store_true');a=p.parse_args();c=config(a.config)
    import torch,onnx,numpy as np
    from model import restore,Core
    checkpoint=ROOT/'artifacts/metal_nut'/('smoke' if a.smoke else 'training')/'model.ckpt'
    model=restore(c,checkpoint).cpu().eval();out=ROOT/'artifacts/metal_nut'/('smoke_export' if a.smoke else 'fp32');out.mkdir(parents=True,exist_ok=True)
    # Official exporter first; post/pre processors disabled intentionally in training.
    full=model.to_onnx(export_root=out/'official',input_size=(256,256))
    core=Core(model).eval();tensor=torch.from_numpy(preprocess(ROOT/verified_records('validation')[0]['path']))
    with torch.no_grad():reference=core(tensor).numpy()
    torch.onnx.export(core,tensor,out/'model.onnx',input_names=['image'],output_names=['raw_anomaly_map'],opset_version=17,dynamo=False)
    graph=onnx.load(out/'model.onnx');onnx.checker.check_model(graph)
    session=ort_session(out/'model.onnx',ROOT/'reports/export_profile');actual=session.run(None,{'image':tensor.numpy()})[0]
    if reference.shape!=actual.shape or not np.isfinite(actual).all():raise ValueError('invalid output shape/values')
    np.testing.assert_allclose(actual,reference,atol=c['alignment']['atol'],rtol=c['alignment']['rtol'])
    for node in graph.graph.node:
        if not node.name:node.name=f'{node.op_type}_{len(node.output)}_{node.output[0]}'
    onnx.save(graph,out/'model.onnx')
    manifest={'schema_version':1,'model_id':'efficientad_s_'+('smoke' if a.smoke else 'fp32'),'category':'metal_nut',
      'checkpoint_sha256':sha256(checkpoint),'onnx_path':'model.onnx','onnx_sha256':sha256(out/'model.onnx'),
      'input':{'name':'image','shape':list(tensor.shape),'dtype':'float32','color':'RGB','layout':'NCHW','range':[0,1]},
      'output':{'name':'raw_anomaly_map','shape':list(reference.shape),'dtype':'float32'},'resize':'opencv_inter_linear',
      'preprocess_owner':'external','normalization_owner':'model','score_reduction':'max','status':'uncalibrated',
      'image_threshold':None,'pixel_threshold':None,'validation_manifest_sha256':None,
      'training_manifest_sha256':sha256(ROOT/'data/train_normal.json'),'stats_manifest_sha256':sha256(ROOT/'data/stats_normal.json'),
      'requested_precision':'FP32','tools':{n:importlib.metadata.version(n) for n in ['anomalib','torch','onnx','onnxruntime-gpu']},'smoke':a.smoke}
    import jsonschema
    from common import read
    jsonschema.validate(manifest,read(ROOT/'schemas/model_manifest.schema.json'));save(out/'model_manifest.json',manifest)
    save(out/'graph_audit.json',{'official_export':str(full),'official_sha256':sha256(full),'opset':[{ 'domain':o.domain,'version':o.version} for o in graph.opset_import],
      'ir_version':graph.ir_version,'operators':dict(Counter(n.op_type for n in graph.graph.node)),
      'normalization':'Anomalib PDN/AE imagenet_norm_batch retained; external divides 255 only',
      'preprocessing':'No external Normalize; fixed OpenCV resize occurs before both torch and ORT',
      'mae':float(np.abs(actual-reference).mean()),'max_abs':float(np.abs(actual-reference).max()),'profile':session.end_profiling()})
if __name__=='__main__':main()
