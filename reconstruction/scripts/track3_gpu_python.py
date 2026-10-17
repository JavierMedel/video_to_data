#!/usr/bin/env python
"""Run a Python GPU module with a scoped PyTorch allocation budget."""
import os
import runpy
import sys


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] != '-m':
        raise SystemExit('Usage: track3_gpu_python.py -m module [module arguments]')
    import torch
    fraction = float(os.environ.get('V2D_TORCH_MEMORY_FRACTION', '0.55'))
    if not 0 < fraction <= 1:
        raise ValueError('V2D_TORCH_MEMORY_FRACTION must be in (0, 1]')
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f'Track3 PyTorch allocator budget: {fraction:.0%} of VRAM; native/TensorRT allocations are additional.', flush=True)
    module = sys.argv[2]
    sys.argv = [module, *sys.argv[3:]]
    runpy.run_module(module, run_name='__main__', alter_sys=True)
