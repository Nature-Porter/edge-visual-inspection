"""Build the frozen QDQ ONNX with detailed TensorRT layer metadata, offline."""
import argparse
import subprocess
from pathlib import Path

from common import ROOT, read, save, sha256
from integration_tests import runtime_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='int8_percentile')
    parser.add_argument('--precision', choices=['int8', 'fp32'], default='int8')
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    package = ROOT / 'artifacts' / 'metal_nut' / args.candidate
    manifest = read(package / 'model_manifest.json')
    model = package / manifest['onnx_path']
    if manifest['onnx_sha256'] != sha256(model):
        raise ValueError('model hash mismatch')
    trt = ROOT / 'third_party' / 'TensorRT-10.9.0.34'
    executable = trt / 'bin' / 'trtexec.exe'
    if not executable.is_file():
        raise FileNotFoundError(executable)
    output_dir = ROOT / 'runs' / ('trt-detailed-' + args.candidate)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = output_dir / 'model.engine'
    layers_path = output_dir / 'layers.json'
    log_path = output_dir / 'build.log'
    command = [str(executable), f'--onnx={model}']
    if args.precision == 'int8':
        command.append('--int8')
    command += [
               '--profilingVerbosity=detailed', f'--saveEngine={engine}',
               f'--exportLayerInfo={layers_path}', '--skipInference']
    env = runtime_env()
    env['PATH'] = str(trt / 'lib') + ';' + str(trt / 'bin') + ';' + env['PATH']
    result = {'candidate': args.candidate, 'requested_precision': args.precision,
              'model_sha256': sha256(model),
              'command': command, 'status': 'blocked', 'int8_execution_verified': False}
    try:
        with log_path.open('w', encoding='utf-8', errors='replace') as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=args.timeout)
        result['exit_code'] = completed.returncode
        result['build_log_sha256'] = sha256(log_path)
        if completed.returncode:
            result['reason'] = 'trtexec failed to build target ONNX; inspect build.log'
        elif not engine.is_file() or not layers_path.is_file():
            result['reason'] = 'trtexec omitted engine or layer export'
        else:
            result['engine_sha256'] = sha256(engine)
            result['layers_sha256'] = sha256(layers_path)
            layer_doc = read(layers_path)
            layers = layer_doc.get('Layers', layer_doc) if isinstance(layer_doc, dict) else layer_doc
            detailed = [layer for layer in layers if isinstance(layer, dict)]
            precision = [layer for layer in detailed if layer.get('Precision') or
                         any(tensor.get('Format/Datatype') for tensor in layer.get('Inputs', []) + layer.get('Outputs', []))]
            int8_compute = [layer for layer in precision
                            if 'conv' in str(layer.get('LayerType', '')).lower()
                            and layer.get('Weights', {}).get('Type') == 'Int8'
                            and any('Int8' in tensor.get('Format/Datatype', '') for tensor in layer.get('Inputs', []))
                            and 'i8i8' in layer.get('TacticName', '').lower()]
            result.update(layer_count=len(layers), detailed_count=len(detailed),
                          precision_count=len(precision), int8_compute_count=len(int8_compute),
                          int8_compute_layers=int8_compute)
            result['status'] = 'built'
            result['reason'] = ('detailed INT8 compute tactics found; runtime output and quality still require validation'
                                if int8_compute else ('FP32 baseline engine built' if args.precision == 'fp32'
                                                      else 'no detailed INT8 compute layer found'))
    except subprocess.TimeoutExpired:
        result['reason'] = f'trtexec build timed out after {args.timeout} seconds'
        result['build_log_sha256'] = sha256(log_path)
    save(output_dir / 'evidence.json', result)
    print(result['reason'])
    return 0 if result.get('int8_compute_count', 0) > 0 or (args.precision == 'fp32' and result['status'] == 'built') else 3


if __name__ == '__main__':
    raise SystemExit(main())
