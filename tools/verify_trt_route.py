"""Fail-closed audit of the independent TensorRT INT8 validation route."""
import re
from pathlib import Path

from common import ROOT, config, read, save, sha256


def seconds(log, pattern):
    match = re.search(pattern, log)
    if not match:
        raise ValueError(f'missing timing: {pattern}')
    return float(match.group(1))


def main():
    name = 'int8_selective_2_trt_compat'
    package = ROOT / 'artifacts' / 'metal_nut' / name
    folder = ROOT / 'runs' / ('trt-detailed-' + name)
    manifest = read(package / 'model_manifest.json')
    rewrite = read(package / 'graph_rewrite.json')
    equivalence = read(package / 'equivalence.json')
    build = read(folder / 'evidence.json')
    runtime = read(folder / 'runtime_evidence.json')
    precision = read(ROOT / 'reports/trt_detailed_exploratory_selective_2.json')
    quality = read(ROOT / 'reports/trt_direct_validation_quality_selective_2.json')
    source = read(ROOT / 'artifacts/metal_nut/int8_selective_2/model_manifest.json')
    if (manifest['onnx_sha256'] != sha256(package / 'model.onnx')
            or source['onnx_sha256'] != manifest['derived_from_model_sha256']
            or equivalence['source_sha256'] != source['onnx_sha256']
            or equivalence['transformed_sha256'] != manifest['onnx_sha256']
            or rewrite['derived_model_sha256'] != manifest['onnx_sha256']
            or equivalence['status'] != 'passed' or equivalence['validation_count'] != 69
            or equivalence['max_abs'] != 0 or equivalence['same_decision'] != 69):
        raise ValueError('derived model provenance or equivalence failed')
    engine = folder / 'model.engine'
    layers = folder / 'layers.json'
    profile = folder / 'runtime_profile.json'
    if (build['model_sha256'] != manifest['onnx_sha256']
            or sha256(engine) != build['engine_sha256']
            or sha256(layers) != build['layers_sha256']
            or sha256(profile) != runtime['profile_sha256']
            or build['engine_sha256'] != runtime['engine_sha256']
            or precision['engine_sha256'] != build['engine_sha256']
            or quality['engine_sha256'] != build['engine_sha256']
            or precision['model_sha256'] != manifest['onnx_sha256']
            or quality['model_sha256'] != manifest['onnx_sha256']):
        raise ValueError('engine, profile or model hash mismatch')
    if (not precision['int8_execution_verified_for_derived_engine']
            or precision['int8_compute_layer_count'] < 1
            or not all(item['runtime_profile_matched'] and item['runtime_time_ms'] > 0
                       for item in precision['int8_compute_layers'])):
        raise ValueError('no actual INT8 computation-layer execution evidence')
    if (quality['status'] != 'measured' or quality['coverage']['ERROR'] != 0
            or quality['coverage']['valid'] != 69 or not quality['validation_quality_gate_pass']
            or quality['data_sha256'] != sha256(ROOT / 'data/validation.json')
            or quality['frozen_threshold'] != source['image_threshold']
            or quality['final_test_used'] or quality['threshold_recalibrated']):
        raise ValueError('validation quality protocol failed')
    for item in quality['predictions']:
        if item['status'] != 'SUCCESS' or item['exit_code'] != 0:
            raise ValueError('non-successful direct TensorRT prediction')
        output_path = Path(item['command'][-1].split('=', 1)[1])
        raw_files = {'input': ROOT / item['command'][2].split('image:', 1)[1],
                     'output': output_path,
                     'log': output_path.with_name(output_path.name.replace('-output.json', '-trtexec.log'))}
        for label, target in raw_files.items():
            if sha256(target) != item[label + '_sha256']:
                raise ValueError(f'{label} raw evidence hash mismatch')
    baseline = read(ROOT / 'artifacts/metal_nut/fp32/validation_quality.json')['metrics']
    limits = config(ROOT / 'configs/metal_nut.yaml')['quality']
    actual = quality['metrics']
    if (baseline['AUROC'] - actual['AUROC'] > limits['max_auroc_drop']
            or actual['FNR'] - baseline['FNR'] > limits['max_fnr_increase']):
        raise ValueError('reported quality gate does not match metrics')
    build_log_path = folder / 'build.log'
    runtime_log_path = folder / 'runtime.log'
    if (sha256(build_log_path) != build['build_log_sha256']
            or sha256(runtime_log_path) != runtime['runtime_log_sha256']):
        raise ValueError('cold/hot timing log hash mismatch')
    build_log = build_log_path.read_text(encoding='utf-8', errors='replace')
    runtime_log = runtime_log_path.read_text(encoding='utf-8', errors='replace')
    source_trt = read(ROOT / 'reports/trt_cached_validation_quality_selective_2.json')
    cpu_fallback = [entry for entry in source_trt['provider_operator_events']
                    if entry['provider'] == 'CPUExecutionProvider']
    benchmark_path = ROOT / 'reports/trt_engine_benchmark_with_transfers.json'
    benchmark = read(benchmark_path) if benchmark_path.is_file() else None
    if benchmark:
        if benchmark.get('data_transfers') != 'enabled; specified input file is copied to the device':
            raise ValueError('benchmark did not transfer the specified validation input')
        for candidate, folder_name in [('fp32', 'fp32'), ('int8_selective_2', name)]:
            item = benchmark['candidates'][candidate]
            bench_engine = ROOT / 'runs' / ('trt-detailed-' + folder_name) / 'model.engine'
            if sha256(bench_engine) != item['engine_sha256'] or len(item['runs']) != 3:
                raise ValueError('benchmark engine or round count mismatch')
            for run in item['runs']:
                if (sha256(ROOT / run['times']) != run['times_sha256']
                        or sha256(ROOT / run['log']) != run['log_sha256']
                        or run['gpu_compute_ms']['count'] < benchmark['iterations_requested']):
                    raise ValueError('benchmark timing evidence mismatch')
        if benchmark['candidates']['int8_selective_2']['engine_sha256'] != build['engine_sha256']:
            raise ValueError('benchmark did not use validated INT8 engine')
    report = {
        'status': 'passed_validation_experiment',
        'model_sha256': manifest['onnx_sha256'],
        'source_model_sha256': source['onnx_sha256'],
        'engine_sha256': build['engine_sha256'],
        'validation_sha256': quality['data_sha256'],
        'int8_execution_verified': True,
        'int8_conv_layer_count': precision['int8_compute_layer_count'],
        'int8_layer_evidence': 'reports/trt_detailed_exploratory_selective_2.json',
        'validation_quality_pass': True,
        'validation_quality_evidence': 'reports/trt_direct_validation_quality_selective_2.json',
        'source_ort_trt_cpu_fallback': cpu_fallback,
        'direct_engine_cpu_ep_fallback': 0,
        'cold_engine_build_seconds': seconds(build_log, r'Engine built in ([0-9.]+) sec'),
        'hot_engine_deserialize_seconds': seconds(runtime_log, r'Engine deserialized in ([0-9.]+) sec'),
        'final_test_used': False,
        'cpp_trt_integration_verified': False,
        'engine_only_speedup_verified': bool(benchmark),
        'full_cpp_pipeline_speedup_verified': False,
        'engine_only_benchmark': 'reports/trt_engine_benchmark_with_transfers.json' if benchmark else None,
        'engine_only_speedup_fp32_over_int8': benchmark['engine_only_speedup_fp32_over_int8'] if benchmark else None,
        'engine_iteration_latency_speedup_fp32_over_int8': benchmark['engine_iteration_latency_speedup_fp32_over_int8'] if benchmark else None,
        'benchmark_fp32_tf32_policy': 'TensorRT default permits TF32; --noTF32 was not passed' if benchmark else None,
        'scope': 'independent TensorRT engine experiment; C++ production detector remains ORT CUDA',
    }
    save(ROOT / 'reports/trt_verified_selective_2.json', report)
    print(report['status'], report['int8_conv_layer_count'], report['validation_quality_pass'])


if __name__ == '__main__':
    main()
