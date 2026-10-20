"""One-GPU-at-a-time native policy job runner with hash-checked recovery."""
import argparse
from pathlib import Path
import sys
from .common import REPO, load, save, sha, new_run, complete, checked_command
from .policy import checkpoint_metadata
from .ownership import validate_outputs, own_output


def validate_native_job(job, report):
    import json
    if job['track'] not in ('track_3', 'track_2_tier1', 'track_2_tier2', 'track_2_tier3'):
        raise ValueError('Unsupported native track')
    mount = REPO/'robotic_grounding' if job['track'].startswith('track_2') else REPO
    output, reference = Path(job['output_dir']).resolve(), Path(job['reference']).resolve()
    output.relative_to(mount); reference.relative_to(mount)
    if not Path(report).resolve().is_relative_to(mount): raise ValueError('Job reports must be inside the native track mount')
    if sha(reference) != job['reference_sha256']: raise ValueError('Job reference changed')
    parent = job.get('parent_checkpoint')
    if parent:
        Path(parent).resolve().relative_to(mount)
        if sha(parent) != job['parent_checkpoint_sha256']: raise ValueError('Parent checkpoint changed')
    if type(job['target_steps']) is not int or job['target_steps'] <= 0 or job['target_steps'] % 1024:
        raise ValueError('Invalid native training budget')
    expected = {'output_dir': '/repo/'+str(output.relative_to(mount)),
                'task.parquet': '/repo/'+str(reference.relative_to(mount)),
                'training.total_environment_steps': job['target_steps']}
    for key, value in expected.items():
        overrides = [v.split('=', 1)[1] for v in job['overrides'] if v.startswith(key+'=')]
        if len(overrides) != 1: raise ValueError('Native job needs one explicit '+key)
        raw = overrides[0]
        try: actual = json.loads(raw)
        except json.JSONDecodeError: actual = raw
        if actual != value: raise ValueError('Native overrides disagree with declared '+key)


def run_jobs(manifest, report, execute=False):
    jobs = load(manifest)
    if not isinstance(jobs, list) or not jobs: raise ValueError('Require a nonempty native training job list')
    validate_outputs(jobs)
    report = Path(report)
    for job in jobs: validate_native_job(job, report)
    signatures = [dict(job) for job in jobs]
    new_run(report, [manifest], {'jobs': signatures})
    if not execute:
        complete(report, 'JOBS_VALIDATED_NOT_EXECUTED', count=len(jobs)); return
    import fcntl
    with (report/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, job in enumerate(jobs):
            with own_output(job):
                _run_job(index, job, report)
        complete(report, 'TRAINED_REQUIRES_EVALUATION', count=len(jobs))


def _run_job(index, job, report):
    if job['track'] not in ('track_3', 'track_2_tier1', 'track_2_tier2', 'track_2_tier3'):
        raise ValueError('Unsupported native track')
    if sha(job['reference']) != job['reference_sha256']: raise ValueError('Job reference changed')
    parent = job.get('parent_checkpoint')
    if parent and sha(parent) != job['parent_checkpoint_sha256']: raise ValueError('Parent checkpoint changed')
    stage = report/f'job_{index:04d}'
    output = Path(job['output_dir']).resolve()
    mount = REPO/'robotic_grounding' if job['track'].startswith('track_2') else REPO
    output.relative_to(mount)
    if not stage.resolve().is_relative_to(mount):
        raise ValueError('Job reports must be inside the native track mount (robotic_grounding for Track 2)')
    stage_manifest = stage/'execution.json'
    new_run(stage, [job['reference'], *([parent] if parent else [])], {'job': job})
    previous = load(stage_manifest)
    if previous['state'] == 'TRAINED':
        if sha(previous['checkpoint']) != previous['checkpoint_sha256']: raise ValueError('Completed checkpoint changed')
        return
    arguments = list(job['overrides'])
    states = sorted(output.glob('state_*.safetensors'), key=lambda p: int(checkpoint_metadata(p)['environment_steps']))
    if states:
        metadata = checkpoint_metadata(states[-1])
        if int(metadata['environment_steps']) >= job['target_steps']:
            policy = states[-1].with_name(states[-1].name.replace('state_', 'policy_'))
            if not policy.exists(): raise ValueError('Completed learner checkpoint lacks actor checkpoint')
            complete(stage, 'TRAINED', checkpoint=str(policy), checkpoint_sha256=sha(policy)); return
        arguments = [v for v in arguments if not v.startswith(('training.resume=', 'training.checkpoint_path=', 'training.initial_policy_path='))]
        arguments += ['training.resume=true', 'training.initial_policy_path=null',
                      'training.checkpoint_path='+__import__('json').dumps('/repo/'+str(states[-1].relative_to(mount)))]
    command = [sys.executable, REPO/'reconstruction/scripts/run_track3_policy_stage.py', '--track', job['track'],
               '--report', stage/'native', 'scripts.train_flash_sac', *arguments]
    complete(stage, 'TRAINING', command=list(map(str, command)))
    try:
        checked_command(command, stage/'training.log')
    except Exception as exc:
        complete(stage, 'STOPPED_FOR_REVIEW', error=str(exc))
        complete(report, 'STOPPED_FOR_REVIEW', active_job=index)
        raise
    policies = sorted(output.glob('policy_*.safetensors'), key=lambda p: int(checkpoint_metadata(p)['environment_steps']))
    if not policies or int(checkpoint_metadata(policies[-1])['environment_steps']) < job['target_steps']:
        raise ValueError('Native training returned without the requested checkpoint budget')
    complete(stage, 'TRAINED', checkpoint=str(policies[-1]), checkpoint_sha256=sha(policies[-1]))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True); p.add_argument('--report', type=Path, required=True)
    p.add_argument('--execute', action='store_true', help='Actually run the validated native GPU jobs')
    a = p.parse_args(); run_jobs(a.manifest, a.report, a.execute)


if __name__ == '__main__': main()


def plan_jobs(specification, output):
    from .policy import training_job
    specification = Path(specification); spec = load(specification)
    records = []
    for item in spec['episodes']:
        reference = Path(item['reference'])
        if sha(reference) != item['reference_sha256']: raise ValueError('Planning reference hash mismatch')
        records.append(training_job(spec['track'], item['episode_index'], reference, item['output_dir'],
                                   spec['target_steps'], spec.get('arm', 'fresh'), item.get('checkpoint'), spec.get('rules'),
                                   spec.get('reward_profile', 'baseline')))
    validate_outputs(records)
    new_run(output, [specification], {'specification_sha256': sha(specification)})
    save(Path(output)/'jobs.json', records)
    complete(output, 'JOBS_PLANNED_NOT_EXECUTED', count=len(records))
    return records


def plan_main():
    p = argparse.ArgumentParser(description='Create native training jobs without executing them')
    p.add_argument('--specification', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); plan_jobs(a.specification, a.output)
