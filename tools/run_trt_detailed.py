"""Execute an already built detailed TensorRT engine on one verified validation image."""
import argparse
import subprocess

from common import ROOT, preprocess, read, save, sha256, verified_records
from integration_tests import runtime_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='int8_percentile_trt_compat')
    args = parser.parse_args()
    folder = ROOT / 'runs' / ('trt-detailed-' + args.candidate)
    engine = folder / 'model.engine'
    evidence = read(folder / 'evidence.json')
    if sha256(engine) != evidence.get('engine_sha256'):
        raise ValueError('engine hash mismatch')
    record = verified_records('validation')[0]
    image = preprocess(ROOT / record['path'])
    input_path = folder / 'input.bin'
    image.tofile(input_path)
    output_path = folder / 'output.json'
    profile_path = folder / 'runtime_profile.json'
    log_path = folder / 'runtime.log'
    trt = ROOT / 'third_party' / 'TensorRT-10.9.0.34'
    command = [str(trt / 'bin/trtexec.exe'), f'--loadEngine={engine}',
               f'--loadInputs=image:{input_path.relative_to(ROOT).as_posix()}', '--iterations=1', '--warmUp=0',
               '--duration=0', '--avgRuns=1', '--dumpProfile', f'--exportOutput={output_path}',
               f'--exportProfile={profile_path}']
    env = runtime_env()
    env['PATH'] = str(trt / 'lib') + ';' + str(trt / 'bin') + ';' + env['PATH']
    with log_path.open('w', encoding='utf-8', errors='replace') as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=120)
    report = {'candidate': args.candidate, 'engine_sha256': sha256(engine),
              'source_image': record['path'], 'source_sha256': record['sha256'],
              'input_sha256': sha256(input_path), 'command': command,
              'exit_code': completed.returncode, 'runtime_log_sha256': sha256(log_path),
              'output_sha256': sha256(output_path) if output_path.is_file() else None,
              'profile_sha256': sha256(profile_path) if profile_path.is_file() else None,
              'status': 'executed' if completed.returncode == 0 and output_path.is_file() and profile_path.is_file() else 'failed',
              'int8_execution_verified': False}
    save(folder / 'runtime_evidence.json', report)
    print(report['status'])
    return 0 if report['status'] == 'executed' else 3


if __name__ == '__main__':
    raise SystemExit(main())
