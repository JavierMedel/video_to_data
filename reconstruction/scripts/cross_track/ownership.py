"""Output ownership is independent of a controller's report directory."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
from .common import load, save, implementation_paths, sha


def identity(job):
    keys = ('track', 'episode_index', 'arm', 'reference_sha256', 'output_dir',
            'parent_checkpoint_sha256', 'target_steps', 'reward_profile', 'overrides')
    payload = {key: job.get(key) for key in keys}
    payload['output_dir'] = str(Path(job['output_dir']).resolve())
    payload['producer_code_sha256'] = {str(p): sha(p) for p in implementation_paths()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_output(output):
    output = Path(output).resolve()
    if any(p in ('baseline', 'policies_short', 'policies_today', 'submission_short', 'submission_today')
           or p.startswith('submission_') for p in output.parts):
        raise ValueError('Preserve baseline and submitted output directories')
    for parent in (output, *output.parents):
        if any((parent/name).exists() for name in ('kaggle_receipt.json', 'upload_manifest.json')):
            raise ValueError('Output belongs to a submission lineage')
    return output


def validate_owner(job):
    output = check_output(job['output_dir'])
    digest = identity(job)
    if job.get('run_identity', digest) != digest:
        raise ValueError('Job identity does not match its configuration')
    marker = output/'.run_identity.json'
    if marker.exists():
        if load(marker) != {'run_identity': digest}:
            raise ValueError('Output directory belongs to another run')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Nonempty output lacks run identity; resume a verified parent into a new directory')
    return output, digest


def validate_outputs(jobs):
    seen = set()
    for job in jobs:
        output, _ = validate_owner(job)
        if output in seen or any(output.is_relative_to(p) or p.is_relative_to(output) for p in seen):
            raise ValueError('Duplicate or overlapping job output directories')
        seen.add(output)


@contextmanager
def own_output(job):
    output = check_output(job['output_dir'])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Outside the output: an empty unclaimed directory stays distinguishable.
    with (output.parent/('.'+output.name+'.run.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output, digest = validate_owner(job)
        output.mkdir(exist_ok=True)
        marker = output/'.run_identity.json'
        if not marker.exists(): save(marker, {'run_identity': digest})
        yield output
