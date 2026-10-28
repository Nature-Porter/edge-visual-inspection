"""A tiny synthetic Conv fixture for engineering tests ONLY."""
from common import ROOT,save,sha256
def make():
    import cv2,numpy as np,onnx
    from onnx import helper as h,TensorProto as T,numpy_helper as nh
    out=ROOT/'tests/fixtures/conv';out.mkdir(parents=True,exist_ok=True)
    weights=np.array([.25,.5,.25],dtype=np.float32).reshape(1,3,1,1)
    graph=h.make_graph([h.make_node('Conv',['image','weight'],['raw_anomaly_map'],name='fixture_conv',kernel_shape=[1,1])],
        'SYNTHETIC_SOFTWARE_TEST_ONLY',[h.make_tensor_value_info('image',T.FLOAT,[1,3,256,256])],[h.make_tensor_value_info('raw_anomaly_map',T.FLOAT,[1,1,256,256])],[nh.from_array(weights,'weight')])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);model.ir_version=10;onnx.checker.check_model(model);onnx.save(model,out/'model.onnx')
    image=np.random.default_rng(42).integers(0,256,(311,317,3),dtype=np.uint8);cv2.imwrite(str(out/'image.png'),image)
    digest=sha256(out/'model.onnx');datahash=sha256(out/'image.png')
    save(out/'model_manifest.json',{'schema_version':1,'model_id':'SYNTHETIC_CONV_NOT_EFFICIENTAD','category':'metal_nut',
        'checkpoint_sha256':digest,'onnx_path':'model.onnx','onnx_sha256':digest,
        'input':{'name':'image','shape':[1,3,256,256],'dtype':'float32','color':'RGB','layout':'NCHW','range':[0,1]},
        'output':{'name':'raw_anomaly_map','shape':[1,1,256,256],'dtype':'float32'},'resize':'opencv_inter_linear',
        'preprocess_owner':'external','normalization_owner':'model','score_reduction':'max','status':'uncalibrated',
        'image_threshold':None,'pixel_threshold':.5,'validation_manifest_sha256':None,'training_manifest_sha256':datahash,'stats_manifest_sha256':datahash,
        'requested_precision':'FP32','tools':{'generator':'make_fixture.py'},'smoke':True,'fixture':True,
        'fixture_note':'Synthetic Conv has no ImageNet normalization; production contract fields used solely to exercise parser.'})
    save(out/'replay.json',{'schema_version':1,'records':[{'path':str(out/'image.png'),'source_image_id':datahash} for _ in range(100)]})
    print(out)
if __name__=='__main__':make()
