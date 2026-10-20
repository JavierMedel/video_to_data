"""Read-only 23-board snapshots and an explicit, conservative upload gate."""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import sys
from .common import KIT, load, save, utc, sha, checked_command


def competitions():
    config = load(KIT/'config/tracks.json')
    return [(track, key, slug, config['metric_display'][key]['higher_is_better'])
            for track, spec in config['tracks'].items() for key, slug in spec.get('competitions', {}).items()]


def serializable(value):
    if isinstance(value, datetime): return value.isoformat()
    if isinstance(value, dict): return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [serializable(v) for v in value]
    if hasattr(value, 'to_dict'): return serializable(value.to_dict())
    if value is None or isinstance(value, (str, bool, int, float)): return value
    return str(value)


def field(row, *names):
    for name in names:
        if row.get(name) is not None: return row[name]
    return None


def timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def snapshot(output, api=None):
    if api is None:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
    output = Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('Use a fresh dated snapshot directory')
    summary = {'observed_utc': utc(), 'competitions': {}}
    for track, key, slug, higher in competitions():
        leaderboard = serializable(api.competition_leaderboard_view(slug, page_size=100))
        submissions = []
        for page in range(1, 101):
            batch = serializable(api.competition_submissions(slug, page_number=page, page_size=100))
            if not isinstance(batch, list): raise ValueError('Unexpected Kaggle submission history response')
            submissions.extend(batch)
            if len(batch) < 100: break
        else: raise ValueError('Submission pagination truncated; cannot establish quota')
        score = lambda row: float(field(row, 'public_score', 'publicScore', 'score'))
        scored = [s for s in submissions if field(s, 'public_score', 'publicScore') not in (None, '')]
        board = leaderboard if isinstance(leaderboard, list) else leaderboard.get('submissions', leaderboard.get('leaderboard', []))
        board = [row for row in board if field(row, 'score', 'public_score', 'publicScore') not in (None, '')]
        board.sort(key=score, reverse=higher)
        ordered = sorted(submissions, key=lambda r: timestamp(field(r, 'date', 'submitted_at', 'submissionDate')), reverse=True)
        best = (max(scored, key=score) if higher else min(scored, key=score)) if scored else None
        record = {'observed_utc': summary['observed_utc'], 'track': track, 'metric': key, 'slug': slug,
                  'higher_is_better': higher, 'leaderboard': leaderboard, 'submissions': submissions,
                  'history_complete': True, 'own_best': best, 'own_latest': ordered[0] if ordered else None,
                  'targets': {str(rank): score(board[rank-1]) if len(board) >= rank else None for rank in (1, 3, 5, 10)}}
        save(output/(slug+'.json'), record)
        summary['competitions'][slug] = {k: record[k] for k in ('track', 'metric', 'own_best', 'own_latest', 'targets')}
    save(output/'summary.json', summary)
    return summary


def quota(history, rules, now=None):
    """Count attempts, including pending/errors; unknown weekly boundaries fail closed."""
    now = now or datetime.now(timezone.utc)
    if not history.get('history_complete'): raise ValueError('Incomplete submission history')
    observed = timestamp(history['observed_utc'])
    if not timedelta(0) <= now-observed <= timedelta(hours=24): raise ValueError('Snapshot is stale or from the future')
    if not rules.get('same_package_all_metrics', True): raise ValueError('This workflow requires a coherent package')
    mode = rules.get('weekly_window', 'unresolved')
    if mode == 'rolling_7_days':
        start = now-timedelta(days=7)
    elif mode == 'calendar':
        if not rules.get('weekly_window_source_quote'):
            raise ValueError('Calendar boundary requires a documented official answer')
        start = timestamp(rules['week_start_utc'])
        if not start <= now < start+timedelta(days=7): raise ValueError('Calendar window does not contain now')
    else:
        raise ValueError('Weekly window remains unresolved; local implementation can proceed, upload cannot')
    attempts = [r for r in history['submissions'] if start <= timestamp(field(r, 'date', 'submitted_at', 'submissionDate')) <= now]
    unlimited_start = rules.get('unlimited_start_utc')
    unlimited = (bool(rules.get('unlimited_source_quote')) and unlimited_start is not None
                 and timestamp(unlimited_start) <= now < timestamp(rules['freeze_utc']))
    if not unlimited and len(attempts) >= 5: raise ValueError('Weekly limit reached for '+history['slug'])
    if now >= timestamp(rules.get('freeze_utc', '2026-11-04T22:00:00+00:00')):
        raise ValueError('Challenge freeze has passed')
    return {'used': len(attempts), 'start_utc': start.isoformat(), 'unlimited': unlimited}


def submit(packet, track, rules_path, snapshots, report, execute=False):
    """Plan/check by default; --execute is required for an actual upload."""
    rules = load(rules_path); report = Path(report)
    manifest, receipt = report/'upload_manifest.json', report/'kaggle_receipt.json'
    digest = sha(packet)
    if receipt.exists():
        if not manifest.exists() or load(manifest)['packet_sha256'] != digest:
            raise ValueError('Receipt belongs to a different packet; never retry this upload')
        command = [sys.executable, KIT/'submit/v2d_submit.py', '--track', track, '--poll-receipt', receipt]
        if execute:
            checked_command(command, report/'poll.log')
            return load(receipt)
        return {'state': 'POLL_EXISTING_RECEIPT', 'command': list(map(str, command))}
    if manifest.exists():
        raise ValueError('Prior upload attempt has no receipt; inspect this ambiguous outcome before retrying')
    checks = {slug: quota(load(Path(snapshots)/(slug+'.json')), rules) for t, _, slug, _ in competitions()
              if t == track or (track == 'track_3' and t == 'track_3_mesh')}
    if not checks: raise ValueError('Unknown track')
    base = [sys.executable, KIT/'submit/v2d_submit.py', Path(packet).resolve(), '--track', track,
            '-m', 'Versioned cross-track improvement; see producing commit in packet']
    if not execute: return {'state': 'UPLOAD_GATE_PASSED_NOT_UPLOADED', 'checks': checks}
    checked_command(base+['--dry-run'], report/'dry_run.log')
    report.mkdir(parents=True, exist_ok=True)
    # An interrupted upload with no readable receipt is ambiguous, never retry automatically.
    with manifest.open('x') as stream:
        import json
        json.dump({'packet_sha256': digest, 'rules_sha256': sha(rules_path), 'track': track,
                   'checks': checks, 'started_utc': utc()}, stream, indent=2)
    checked_command(base+['--wait', '--receipt', receipt], report/'upload.log')
    return load(receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__); commands = p.add_subparsers(dest='action', required=True)
    snap = commands.add_parser('snapshot'); snap.add_argument('--output', type=Path, required=True)
    upload = commands.add_parser('submit')
    for name in ('packet', 'rules', 'snapshots', 'report'): upload.add_argument('--'+name, type=Path, required=True)
    upload.add_argument('--track', required=True); upload.add_argument('--execute', action='store_true')
    a = p.parse_args()
    result = snapshot(a.output) if a.action == 'snapshot' else submit(a.packet, a.track, a.rules, a.snapshots, a.report, a.execute)
    print(__import__('json').dumps(result, indent=2))


if __name__ == '__main__': main()
