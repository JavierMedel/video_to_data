#!/usr/bin/env python3
"""Dispatch implementation tools; no command launches training/upload by default."""
import argparse
import importlib


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('tool', choices=['checkpoint-determinism', 'tier2-camera', 'track1-evidence', 'surrogate-study', 'tier2-noise', 'condition-tier', 'mhr-inventory', 'mhr-export', 'mhr-parity', 'object-fit', 'human-fit', 'assemble-track1', 'pack-selected', 'diagnostics', 'checkpoint-inventory', 'checkpoint-select', 'prepare-tier', 'tier-inventory', 'plan-jobs', 'track3', 'jobs', 'submission'])
    import sys
    if len(sys.argv) == 1 or sys.argv[1] in ('-h', '--help'):
        p.print_help(); return
    a, remainder = p.parse_known_args()
    modules = {'checkpoint-determinism': 'policy', 'tier2-camera': 'tier2_camera', 'track1-evidence': 'track1_evidence', 'surrogate-study': 'surrogate', 'tier2-noise': 'tier2_noise', 'condition-tier': 'tier_inputs', 'mhr-inventory': 'mhr_asset', 'mhr-export': 'mhr_asset', 'mhr-parity': 'mhr_asset', 'object-fit': 'track1', 'human-fit': 'human', 'assemble-track1': 'packaging', 'pack-selected': 'packaging', 'diagnostics': 'diagnostics', 'checkpoint-inventory': 'policy', 'checkpoint-select': 'policy',
               'prepare-tier': 'tier_inputs', 'tier-inventory': 'tier_inputs', 'plan-jobs': 'jobs', 'track3': 'track3', 'jobs': 'jobs', 'submission': 'submission'}
    sys.argv = [sys.argv[0]+' '+a.tool, *remainder]
    entrypoint = {'checkpoint-determinism': 'determinism_main', 'condition-tier': 'condition_main', 'tier-inventory': 'inventory_main', 'checkpoint-inventory': 'inventory_main', 'plan-jobs': 'plan_main', 'pack-selected': 'policy_main'}.get(a.tool, 'main')
    getattr(importlib.import_module('cross_track.'+modules[a.tool]), entrypoint)()


if __name__ == '__main__': main()
