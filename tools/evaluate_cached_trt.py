"""Validation-only quality of the frozen QDQ model through its cached TRT provider engine."""
import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np

from common import ROOT, preprocess, read, save, sha256, verified_records
from integration_tests import runtime_env
from metrics import join_predictions, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='int8_percentile')
    parser.add_argument('--evidence', default='reports/trt_int8_evidence.json')
    parser.add_argument('--output', default='reports/trt_cached_validation_quality.json')
    args = parser.parse_args()
    pointer = read(ROOT / args.evidence)
    source_evidence = read(Path(pointer['run_dir']) / 'evidence.json')
    package = ROOT / 'artifacts' / 'metal_nut' / args.candidate
    manifest = read(package / 'model_manifest.json')
    model = package / manifest['onnx_path']
    if (source_evidence['key']['model_sha256'] != sha256(model)
            or sha256(model) != manifest['onnx_sha256']):
        raise ValueError('cached TRT model hash mismatch')
    cache = Path(pointer['run_dir']) / 'engine_cache'
    engines = list(cache.glob('*.engine'))
    if len(engines) != 1 or sha256(engines[0]) != source_evidence['cached_engines'][0]['sha256']:
        raise ValueError('cached engine hash mismatch')
    trt = ROOT / 'third_party' / 'TensorRT-10.9.0.34'
    env = runtime_env()
    os.environ['PATH'] = str(trt / 'lib') + ';' + str(trt / 'bin') + ';' + env['PATH']
    handles = []
    for folder in os.environ['PATH'].split(';'):
        if Path(folder).is_dir():
            try:
                handles.append(os.add_dll_directory(folder))
            except OSError:
                pass
    import onnxruntime as ort
    output = ROOT / 'runs' / ('trt-cached-validation-' + args.candidate)
    output.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(output / 'ort-profile')
    session = ort.InferenceSession(str(model), options, providers=[
        ('TensorrtExecutionProvider', {'trt_engine_cache_enable': True,
                                      'trt_engine_cache_path': str(cache)}),
        ('CUDAExecutionProvider', {'use_tf32': False}), 'CPUExecutionProvider'])
    session.disable_fallback()
    rows = verified_records('validation')
    predictions = []
    for row in rows:
        try:
            value = session.run([manifest['output']['name']],
                                {manifest['input']['name']: preprocess(ROOT / row['path'])})[0]
            if list(value.shape) != manifest['output']['shape'] or not np.isfinite(value).all():
                raise ValueError('invalid output')
            predictions.append({'source_image_id': row['source_image_id'],
                                'score': float(value.max()), 'status': 'SUCCESS'})
        except Exception as exc:
            predictions.append({'source_image_id': row['source_image_id'],
                                'score': None, 'status': 'ERROR', 'reason': str(exc)})
    profile = Path(session.end_profiling())
    nodes = [item['args'] for item in read(profile)
             if item.get('cat') == 'Node' and item.get('args', {}).get('provider')]
    providers = Counter((item['provider'], item.get('op_name', 'unknown')) for item in nodes)
    valid, coverage = join_predictions(rows, predictions)
    source_predictions = read(package / 'validation_predictions.json')
    paired = {item['source_image_id']: item['score'] for item in source_predictions}
    score_max_abs = max(abs(item['score'] - paired[item['source_image_id']])
                        for item in predictions if item['status'] == 'SUCCESS')
    report = {'candidate': args.candidate, 'model_sha256': sha256(model),
              'engine_sha256': sha256(engines[0]), 'data_sha256': sha256(ROOT / 'data/validation.json'),
              'profile': str(profile), 'profile_sha256': sha256(profile),
              'provider_operator_events': [dict(provider=provider, operator=operator, events=count)
                                           for (provider, operator), count in sorted(providers.items())],
              'coverage': coverage, 'frozen_threshold': manifest['image_threshold'],
              'max_score_abs_vs_ort_cuda': score_max_abs,
              'metrics': metrics(*zip(*valid), manifest['image_threshold']) if not coverage['ERROR'] else None,
              'status': 'measured' if not coverage['ERROR'] else 'failed',
              'final_test_used': False, 'threshold_recalibrated': False,
              'int8_execution_verified': False}
    save(output / 'validation_predictions.json', predictions)
    save(ROOT / args.output, report)
    print(report['status'], report['coverage'], report['metrics'])
    return 0 if report['status'] == 'measured' else 5


if __name__ == '__main__':
    raise SystemExit(main())
