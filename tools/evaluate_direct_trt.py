"""Validation-only quality through an already built direct TensorRT INT8 engine."""
import argparse
import subprocess

import numpy as np

from common import ROOT, config, preprocess, read, save, sha256, verified_records
from integration_tests import runtime_env
from metrics import join_predictions, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='int8_selective_2_trt_compat')
    parser.add_argument('--output', default='reports/trt_direct_validation_quality_selective_2.json')
    args = parser.parse_args()
    package = ROOT / 'artifacts' / 'metal_nut' / args.candidate
    manifest = read(package / 'model_manifest.json')
    equivalent = read(package / 'equivalence.json')
    if (manifest['status'] != 'calibrated' or equivalent['status'] != 'passed'
            or equivalent['transformed_sha256'] != manifest['onnx_sha256']
            or sha256(package / 'model.onnx') != manifest['onnx_sha256']):
        raise ValueError('derived graph lacks validated threshold or hash')
    folder = ROOT / 'runs' / ('trt-detailed-' + args.candidate)
    engine = folder / 'model.engine'
    build = read(folder / 'evidence.json')
    execution = read(ROOT / 'reports/trt_detailed_exploratory_selective_2.json')
    if (sha256(engine) != build['engine_sha256'] or build['model_sha256'] != manifest['onnx_sha256']
            or execution['engine_sha256'] != build['engine_sha256']
            or not execution['int8_execution_verified_for_derived_engine']):
        raise ValueError('engine precision or model evidence mismatch')
    records = verified_records('validation')
    raw_dir = folder / 'validation'
    raw_dir.mkdir(parents=True, exist_ok=True)
    trt = ROOT / 'third_party' / 'TensorRT-10.9.0.34'
    env = runtime_env()
    env['PATH'] = str(trt / 'lib') + ';' + str(trt / 'bin') + ';' + env['PATH']
    predictions = []
    commands = []
    for index, record in enumerate(records):
        stem = f'{index:03d}'
        image = preprocess(ROOT / record['path'])
        input_path = raw_dir / (stem + '-input.bin')
        output_path = raw_dir / (stem + '-output.json')
        log_path = raw_dir / (stem + '-trtexec.log')
        image.tofile(input_path)
        command = [str(trt / 'bin/trtexec.exe'), f'--loadEngine={engine}',
                   f'--loadInputs=image:{input_path.relative_to(ROOT).as_posix()}',
                   '--iterations=1', '--warmUp=0', '--duration=0', '--avgRuns=1',
                   f'--exportOutput={output_path}']
        with log_path.open('w', encoding='utf-8', errors='replace') as log:
            try:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=120)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = 124
        item = {'source_image_id': record['source_image_id'], 'source_sha256': record['sha256'],
                'input_sha256': sha256(input_path), 'log_sha256': sha256(log_path),
                'output_sha256': sha256(output_path) if output_path.is_file() else None,
                'exit_code': code, 'command': command}
        if code == 0 and output_path.is_file():
            output = read(output_path)
            if (len(output) == 1 and output[0]['name'] == manifest['output']['name']
                    and len(output[0]['values']) == 256 * 256):
                values = np.asarray(output[0]['values'], dtype=np.float32)
                if np.isfinite(values).all():
                    item.update(status='SUCCESS', score=float(values.max()))
                else:
                    item.update(status='ERROR', score=None, reason='nonfinite output')
            else:
                item.update(status='ERROR', score=None, reason='output contract mismatch')
        else:
            item.update(status='ERROR', score=None, reason=f'trtexec exit {code}')
        predictions.append(item)
        commands.append(command)
        print(f'{index + 1}/{len(records)} {item["status"]}', flush=True)
    valid, coverage = join_predictions(records, predictions)
    source_name = args.candidate.removesuffix('_trt_compat')
    source_preds = read(ROOT / 'artifacts' / 'metal_nut' / source_name / 'validation_predictions.json')
    source_scores = {item['source_image_id']: item['score'] for item in source_preds}
    max_score_abs = max((abs(item['score'] - source_scores[item['source_image_id']])
                         for item in predictions if item['status'] == 'SUCCESS'), default=None)
    result = metrics(*zip(*valid), manifest['image_threshold']) if not coverage['ERROR'] else None
    baseline = read(ROOT / 'artifacts/metal_nut/fp32/validation_quality.json')['metrics']
    gates = config(ROOT / 'configs/metal_nut.yaml')['quality']
    gate = (result is not None
            and baseline['AUROC'] - result['AUROC'] <= gates['max_auroc_drop']
            and result['FNR'] - baseline['FNR'] <= gates['max_fnr_increase'])
    report = {'candidate': args.candidate, 'model_sha256': manifest['onnx_sha256'],
              'source_model_sha256': manifest['derived_from_model_sha256'],
              'engine_sha256': build['engine_sha256'],
              'int8_layer_evidence': 'reports/trt_detailed_exploratory_selective_2.json',
              'data_sha256': sha256(ROOT / 'data/validation.json'), 'coverage': coverage,
              'metrics': result, 'validation_quality_gate_pass': gate,
              'max_score_abs_vs_source_ort_cuda': max_score_abs,
              'frozen_threshold': manifest['image_threshold'],
              'threshold_recalibrated': False, 'final_test_used': False,
              'raw_dir': str(raw_dir.relative_to(ROOT)), 'predictions': predictions,
              'status': 'measured' if not coverage['ERROR'] else 'failed'}
    save(ROOT / args.output, report)
    print(report['status'], report['metrics'], 'gate', gate)
    return 0 if report['status'] == 'measured' else 5


if __name__ == '__main__':
    raise SystemExit(main())
