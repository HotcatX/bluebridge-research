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
import xml.etree.ElementTree as ET


def validate_svg_viewport(path):
    """Reject the high-DPI Graphviz SVG scaling mismatch before publication."""
    root = ET.parse(path).getroot()
    graph = root.find('{http://www.w3.org/2000/svg}g')
    if graph is None or not graph.get('transform', '').startswith('scale(1 1) '):
        raise ValueError(f'{path.name}: SVG coordinates must use unit scale to avoid clipping')


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
            # Graphviz 8 may enlarge the SVG graph without enlarging viewBox
            # when source dpi=150. SVG is resolution independent; use its
            # native 72-DPI coordinates and reserve 150 DPI for raster output.
            dpi = '72' if fmt == 'svg' else '150'
            subprocess.run([args.dot, '-Gdpi=' + dpi, '-T' + fmt,
                            str(source), '-o', str(destination)], check=True)
            if fmt == 'svg':
                validate_svg_viewport(destination)
            print(f'{destination.relative_to(directory.parent.parent)} ({destination.stat().st_size:,} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
