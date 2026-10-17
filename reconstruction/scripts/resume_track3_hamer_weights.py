#!/usr/bin/env python3
"""Resume the upstream HaMeR demo archive without discarding a partial download."""
import datetime
import gzip
import json
import os
from pathlib import Path
import tarfile
import time
import urllib.request

from prepare_track3_campaign import ROOT, sha256
from run_track3_object_pilot import save

URL = 'https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz'


def main():
    directory = ROOT / 'data/weights/hamer'
    report_path = ROOT / 'incident_reports/track3_baseline/hamer_provisioning.json'
    archive = directory / 'hamer_demo_data.tar.gz'
    directory.mkdir(parents=True, exist_ok=True)
    # Recover a possibly unflushed tail after the host crash. Preserve the exact
    # old tail so this repair is reviewable and only re-download the last 8 MiB.
    if archive.exists() and archive.stat().st_size:
        old_size = archive.stat().st_size
        rollback = min(old_size, 8 * 1024 * 1024)
        backup = directory / f'interrupted_tail_{time.time_ns()}.bin'
        with archive.open('r+b') as stream:
            stream.seek(old_size - rollback); backup.write_bytes(stream.read())
            stream.truncate(old_size - rollback)
    start = archive.stat().st_size if archive.exists() else 0
    state = {'state': 'DOWNLOADING', 'url': URL, 'resumed_from_bytes': start,
             'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save(report_path, state)
    request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-'} if start else {})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if start and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-')):
                raise RuntimeError('Server did not honor the resume range; partial archive preserved')
            total = start + int(response.headers['Content-Length'])
            state.update(expected_total_bytes=total, etag=response.headers.get('ETag'))
            print('RESUME', start, 'of', total, 'bytes', flush=True)
            last = time.monotonic()
            with archive.open('ab') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    if time.monotonic() - last > 10:
                        output.flush(); os.fsync(output.fileno())
                        state['downloaded_bytes'] = output.tell(); save(report_path, state)
                        print('DOWNLOADED', output.tell(), 'of', total, flush=True)
                        last = time.monotonic()
                output.flush(); os.fsync(output.fileno())
        if archive.stat().st_size != total:
            raise RuntimeError('Incomplete archive; restart this command to resume')
        state.update(state='VERIFYING_GZIP', downloaded_bytes=total); save(report_path, state)
        with gzip.open(archive, 'rb') as stream:
            while stream.read(1024 * 1024):
                pass
        state['archive_sha256'] = sha256(archive)
        state['state'] = 'EXTRACTING'; save(report_path, state)
        with tarfile.open(archive, 'r|gz') as bundle:
            bundle.extractall(directory, filter='data')
        checkpoint = directory / '_DATA/hamer_ckpts/checkpoints/hamer.ckpt'
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError('Expected HaMeR checkpoint missing after extraction')
        state.update(state='COMPLETE', checkpoint_sha256=sha256(checkpoint), checkpoint=str(checkpoint.relative_to(ROOT)))
        print('COMPLETE', checkpoint, flush=True)
    except Exception as exc:
        state.update(state='INTERRUPTED_OR_FAILED', error=str(exc))
        raise
    finally:
        state['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(report_path, state)


if __name__ == '__main__':
    main()
