"""Make an exploratory TensorRT-compatible copy by folding constant bias DQ nodes."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from common import ROOT, read, save, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='int8_percentile')
    args = parser.parse_args()
    source_dir = ROOT / 'artifacts' / 'metal_nut' / args.candidate
    source_manifest = read(source_dir / 'model_manifest.json')
    source = source_dir / source_manifest['onnx_path']
    if sha256(source) != source_manifest['onnx_sha256']:
        raise ValueError('source model hash mismatch')
    model = onnx.load(source)
    initializers = {item.name: item for item in model.graph.initializer}
    removable = []
    folded = []
    for node in model.graph.node:
        if node.op_type != 'DequantizeLinear' or '.bias' not in node.name:
            continue
        if len(node.input) != 3 or any(name not in initializers for name in node.input):
            raise ValueError(f'nonconstant bias DQ: {node.name}')
        q, scale, zero = (numpy_helper.to_array(initializers[name]) for name in node.input)
        if q.ndim != 1 or scale.shape != q.shape or zero.shape != q.shape:
            raise ValueError(f'unexpected bias DQ shape: {node.name}')
        value = ((q.astype(np.float32) - zero.astype(np.float32)) * scale.astype(np.float32)).astype(np.float32)
        model.graph.initializer.append(numpy_helper.from_array(value, name=node.output[0]))
        removable.append(node)
        folded.append({'node': node.name, 'length': int(q.size), 'output': node.output[0]})
    if not folded:
        raise ValueError('no constant bias DQ nodes found')
    for node in removable:
        model.graph.node.remove(node)
    output_dir = ROOT / 'artifacts' / 'metal_nut' / (args.candidate + '_trt_compat')
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / 'model.onnx'
    onnx.save(model, target)
    onnx.checker.check_model(str(target))
    manifest = deepcopy(source_manifest)
    manifest['model_id'] += '_trt_compat'
    manifest['onnx_sha256'] = sha256(target)
    manifest['status'] = 'uncalibrated'
    manifest['image_threshold'] = None
    manifest['derived_from_model_sha256'] = source_manifest['onnx_sha256']
    manifest['graph_rewrite'] = f'fold {len(folded)} constant bias DequantizeLinear nodes to float32 initializers; validation equivalence required'
    save(output_dir / 'model_manifest.json', manifest)
    save(output_dir / 'graph_rewrite.json', {
        'source_model_sha256': source_manifest['onnx_sha256'],
        'derived_model_sha256': sha256(target), 'folded': folded,
        'status': 'exploratory_unvalidated',
        'final_test_used': False,
    })
    print(target)


if __name__ == '__main__':
    main()
