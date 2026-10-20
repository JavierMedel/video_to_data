"""Artifact contracts shared by the cross-track improvement tools."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
KIT = REPO / 'reconstruction/v2d_submission_kit/v2d_submission_kit'


def artifact_path(path):
    """Rebase full-repository Docker receipts when reviewing them on the host."""
    path = Path(path)
    if not path.exists() and path.is_absolute() and path.is_relative_to('/repo') and REPO != Path('/repo'):
        return REPO/path.relative_to('/repo')
    return path


def sha(path):
    with artifact_path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def input_digest(record, path):
    expected = artifact_path(path).resolve()
    matches = [digest for source, digest in record['inputs_sha256'].items()
               if artifact_path(source).resolve() == expected]
    if len(matches) != 1: raise ValueError('Review lacks one unambiguous parent input')
    return matches[0]


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def kit_imports():
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))


def metric(track, name):
    """Load the checked-in official metric module without modifying its source."""
    path = KIT / 'metric_code' / track / (name + '.py')
    spec = importlib.util.spec_from_file_location('official_' + track + '_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def implementation_paths():
    paths = sorted(Path(__file__).parent.glob('*.py'))
    paths += [Path(__file__).parents[1]/name for name in ('run_cross_track_improvement.py', 'track1_metric_harness.py', 'run_track3_policy_stage.py')]
    native = REPO/'robotic_grounding/flash_chord'
    paths += [native/name for name in ('scripts/evaluate_policy.py', 'src/flash_chord/training/flash_sac/runner.py',
                                      'src/flash_chord/training/flash_sac/config.py',
                                      'src/flash_chord/configs/training/reference_flash_sac.yaml')]
    return paths


def new_run(output, inputs, configuration):
    """An existing version may resume only with identical inputs/configuration."""
    output = Path(output).resolve()
    paths = {str(Path(p).resolve()): sha(p) for p in inputs}
    if any(output == Path(p) or output in Path(p).parents for p in paths):
        raise ValueError('Output must not contain an immutable input')
    implementation = implementation_paths()
    signature = {'inputs_sha256': paths, 'configuration': configuration,
                 'implementation_sha256': {str(p.resolve()): sha(p) for p in implementation}}
    manifest = output / 'execution.json'
    if manifest.exists():
        previous = load(manifest)
        if any(previous.get(k) != v for k, v in signature.items()):
            raise ValueError('Existing version has different inputs/config; choose a new version')
        return previous
    if output.exists() and any(output.iterdir()):
        raise ValueError('Nonempty version directory lacks an execution manifest')
    record = {**signature, 'state': 'PREPARED', 'created_utc': utc(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
              'uploaded': False}
    save(manifest, record)
    return record


def complete(output, state, **fields):
    path = Path(output) / 'execution.json'
    record = load(path)
    record.update(state=state, observed_utc=utc(), **fields)
    save(path, record)
    return record


def checked_command(command, log):
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with Path(log).open('a') as stream:
        stream.write(json.dumps({'command': list(map(str, command)), 'utc': utc()}) + '\n')
        stream.flush()
        subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, check=True)
