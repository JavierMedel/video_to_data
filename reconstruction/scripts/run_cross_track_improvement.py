#!/usr/bin/env python3
"""Dispatch implementation tools; no command launches training/upload by default."""
import argparse
import importlib


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('tool', choices=['object-fit', 'human-fit', 'assemble-track1', 'pack-selected', 'diagnostics', 'checkpoint-inventory', 'checkpoint-select', 'prepare-tier', 'tier-inventory', 'plan-jobs', 'track3', 'jobs', 'submission'])
    import sys
    if len(sys.argv) == 1 or sys.argv[1] in ('-h', '--help'):
        p.print_help(); return
    a, remainder = p.parse_known_args()
    modules = {'object-fit': 'track1', 'human-fit': 'human', 'assemble-track1': 'packaging', 'pack-selected': 'packaging', 'diagnostics': 'diagnostics', 'checkpoint-inventory': 'policy', 'checkpoint-select': 'policy',
               'prepare-tier': 'tier_inputs', 'tier-inventory': 'tier_inputs', 'plan-jobs': 'jobs', 'track3': 'track3', 'jobs': 'jobs', 'submission': 'submission'}
    sys.argv = [sys.argv[0]+' '+a.tool, *remainder]
    entrypoint = {'tier-inventory': 'inventory_main', 'checkpoint-inventory': 'inventory_main', 'plan-jobs': 'plan_main', 'pack-selected': 'policy_main'}.get(a.tool, 'main')
    getattr(importlib.import_module('cross_track.'+modules[a.tool]), entrypoint)()


if __name__ == '__main__': main()
