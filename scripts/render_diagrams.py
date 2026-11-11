#!/usr/bin/env python3
"""Render the checked-in Graphviz diagrams; requires the external `dot` command.

No model, dataset, network or remote compute is used. Run from any directory:
    python3 scripts/render_diagrams.py
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dot', default=shutil.which('dot'), help='Graphviz dot executable')
    args = parser.parse_args()
    if not args.dot:
        parser.error('Graphviz is required: install it and place dot on PATH, or pass --dot')
    directory = Path(__file__).resolve().parents[1] / 'docs' / 'figures'
    for source in sorted(directory.glob('*.dot')):
        for fmt in ('svg', 'png'):
            destination = source.with_suffix('.' + fmt)
            subprocess.run([args.dot, '-T' + fmt, str(source), '-o', str(destination)], check=True)
            print(f'{destination.relative_to(directory.parent.parent)} ({destination.stat().st_size:,} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
