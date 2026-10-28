"""Join target hash, detailed INT8 tactics, runtime profile and numerical audit."""
import argparse

import numpy as np

from common import ROOT, ort_session, preprocess, read, save, sha256, verified_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='int8_percentile_trt_compat')
    parser.add_argument('--output', default='reports/trt_detailed_exploratory.json')
    args = parser.parse_args()
    package = ROOT / 'artifacts' / 'metal_nut' / args.candidate
    manifest = read(package / 'model_manifest.json')
    equivalence = read(package / 'equivalence.json')
    if (equivalence['status'] != 'passed' or equivalence['transformed_sha256'] != manifest['onnx_sha256']
            or sha256(package / 'model.onnx') != manifest['onnx_sha256']):
        raise ValueError('derived graph equivalence or model hash missing')
    folder = ROOT / 'runs' / ('trt-detailed-' + args.candidate)
    build = read(folder / 'evidence.json')
    runtime = read(folder / 'runtime_evidence.json')
    checks = {'engine': (folder / 'model.engine', build['engine_sha256']),
              'layers': (folder / 'layers.json', build['layers_sha256']),
              'runtime_log': (folder / 'runtime.log', runtime['runtime_log_sha256']),
              'runtime_profile': (folder / 'runtime_profile.json', runtime['profile_sha256']),
              'output': (folder / 'output.json', runtime['output_sha256'])}
    for label, (path, digest) in checks.items():
        if not digest or sha256(path) != digest:
            raise ValueError(f'{label} hash mismatch')
    layers = read(folder / 'layers.json')['Layers']
    profile = {item.get('name'): item for item in read(folder / 'runtime_profile.json') if item.get('name')}
    int8 = []
    for layer in layers:
        if ('conv' not in layer.get('LayerType', '').lower()
                or layer.get('Weights', {}).get('Type') != 'Int8'
                or not any('Int8' in tensor.get('Format/Datatype', '') for tensor in layer.get('Inputs', []))
                or 'i8i8' not in layer.get('TacticName', '').lower()):
            continue
        timing = profile.get(layer['Name'])
        int8.append({'name': layer['Name'], 'layer_type': layer['LayerType'],
                     'weights_type': layer['Weights']['Type'],
                     'input_formats': [item.get('Format/Datatype') for item in layer.get('Inputs', [])],
                     'output_formats': [item.get('Format/Datatype') for item in layer.get('Outputs', [])],
                     'tactic': layer['TacticName'],
                     'runtime_profile_matched': timing is not None,
                     'runtime_time_ms': timing.get('timeMs') if timing else None})
    int8_executed = (runtime['status'] == 'executed' and len(int8) > 0
                     and all(item['runtime_profile_matched'] and item['runtime_time_ms'] > 0 for item in int8))
    record = verified_records('validation')[0]
    if record['sha256'] != runtime['source_sha256'] or record['path'] != runtime['source_image']:
        raise ValueError('runtime image mismatch')
    image = preprocess(ROOT / record['path'])
    session = ort_session(package / 'model.onnx')
    reference = session.run([manifest['output']['name']], {manifest['input']['name']: image})[0]
    exported = read(folder / 'output.json')
    if len(exported) != 1 or exported[0]['name'] != manifest['output']['name']:
        raise ValueError('unexpected TensorRT output binding')
    actual = np.asarray(exported[0]['values'], dtype=np.float32).reshape(reference.shape)
    if not np.isfinite(actual).all():
        raise ValueError('nonfinite TensorRT output')
    max_abs = float(np.max(np.abs(actual - reference)))
    score_abs = abs(float(np.max(actual)) - float(np.max(reference)))
    report = {'candidate': args.candidate, 'model_sha256': manifest['onnx_sha256'],
              'derived_from_sha256': manifest['derived_from_model_sha256'],
              'engine_sha256': build['engine_sha256'], 'layer_export_sha256': build['layers_sha256'],
              'runtime_profile_sha256': runtime['profile_sha256'], 'output_sha256': runtime['output_sha256'],
              'int8_compute_layer_count': len(int8), 'int8_compute_layers': int8,
              'int8_execution_verified_for_derived_engine': int8_executed,
              'reference_validation_image': record['path'],
              'trt_vs_ort_max_abs': max_abs, 'trt_vs_ort_score_abs': score_abs,
              'trt_vs_ort_allclose_atol_1e_5_rtol_1e_4': bool(np.allclose(actual, reference, atol=1e-5, rtol=1e-4)),
              'quality_accepted': False, 'final_test_used_for_derived_engine': False,
              'note': 'This is an exploratory graph rewrite and direct TensorRT engine, not the frozen C++ ORT CUDA target.'}
    save(ROOT / args.output, report)
    print('INT8 executed:', int8_executed, 'compute layers:', len(int8), 'max abs:', max_abs)
    return 0 if int8_executed else 3


if __name__ == '__main__':
    raise SystemExit(main())
