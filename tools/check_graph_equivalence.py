"""Check an exploratory bias-folded graph against its frozen source on validation only."""
import argparse

import numpy as np

from common import ROOT, ort_session, preprocess, read, save, sha256, verified_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='int8_percentile')
    args = parser.parse_args()
    source_dir = ROOT / 'artifacts' / 'metal_nut' / args.candidate
    target_dir = ROOT / 'artifacts' / 'metal_nut' / (args.candidate + '_trt_compat')
    source = read(source_dir / 'model_manifest.json')
    target = read(target_dir / 'model_manifest.json')
    if target['derived_from_model_sha256'] != source['onnx_sha256']:
        raise ValueError('source provenance mismatch')
    original_model = source_dir / source['onnx_path']
    transformed_model = target_dir / target['onnx_path']
    if sha256(original_model) != source['onnx_sha256'] or sha256(transformed_model) != target['onnx_sha256']:
        raise ValueError('model hash mismatch')
    original = ort_session(original_model)
    transformed = ort_session(transformed_model)
    max_abs = 0.0
    max_score_abs = 0.0
    same_decision = 0
    close_count = 0
    records = verified_records('validation')
    for record in records:
        image = preprocess(ROOT / record['path'])
        first = original.run([source['output']['name']], {source['input']['name']: image})[0]
        second = transformed.run([target['output']['name']], {target['input']['name']: image})[0]
        if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError('shape or finiteness mismatch')
        max_abs = max(max_abs, float(np.max(np.abs(first - second))))
        score_first, score_second = float(np.max(first)), float(np.max(second))
        max_score_abs = max(max_score_abs, abs(score_first - score_second))
        same_decision += (score_first > source['image_threshold']) == (score_second > source['image_threshold'])
        close_count += int(np.allclose(first, second, atol=1e-5, rtol=1e-4))
    passed = close_count == len(records) and same_decision == len(records)
    report = {'source_sha256': source['onnx_sha256'], 'transformed_sha256': target['onnx_sha256'],
              'validation_count': len(records), 'same_tensor_within_tolerance': close_count,
              'same_decision': same_decision, 'max_abs': max_abs, 'max_score_abs': max_score_abs,
              'atol': 1e-5, 'rtol': 1e-4, 'final_test_used': False,
              'status': 'passed' if passed else 'failed'}
    save(target_dir / 'equivalence.json', report)
    if passed:
        target['status'] = 'calibrated'
        target['image_threshold'] = source['image_threshold']
        save(target_dir / 'model_manifest.json', target)
    print(report)
    return 0 if passed else 5


if __name__ == '__main__':
    raise SystemExit(main())
