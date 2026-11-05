"""Shared command-line entry point for the ``generate_processed_dataset*.py`` scripts.

Each script defines ``generate_<name>(dataset_path, out_prefix, ...)`` functions. This module
turns them into a CLI::

    python generate_processed_dataset.py --dataset rat7m --in /data/raw/rat7m --out /data/processed

``--in`` is the raw dataset folder itself; the output goes to ``<--out>/<dataset name>/``.
"""

import argparse
import inspect
import os
from pathlib import Path

#: keypoint scheme yaml files shipped with the package
KEYPOINTS_DIR = str(Path(__file__).resolve().parent / 'keypoints')


def collect_generators(namespace):
    """Map dataset name -> generate_* function, using each function's ``dataset_name`` default."""
    gens = {}
    for fname, fn in sorted(namespace.items()):
        if not (fname.startswith('generate_') and inspect.isfunction(fn)):
            continue
        param = inspect.signature(fn).parameters.get('dataset_name')
        if param is None or param.default is inspect.Parameter.empty:
            continue
        name = param.default
        assert name not in gens, f'duplicate dataset name {name!r} ({fname}, {gens[name].__name__})'
        gens[name] = fn
    return gens


def main(generators, argv=None, description=None):
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='datasets:\n  ' + '\n  '.join(
            f'{name:<22s} {(fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""}'
            for name, fn in generators.items()))
    ap.add_argument('--dataset', required=True, choices=sorted(generators),
                    metavar='NAME', help='dataset to generate (see list below)')
    ap.add_argument('--in', dest='raw', required=True, metavar='RAW_DIR',
                    help='raw dataset folder (as downloaded from the source)')
    ap.add_argument('--out', required=True, metavar='OUT_ROOT',
                    help='output root; data is written to OUT_ROOT/<dataset name>/')
    ap.add_argument('--name', default=None,
                    help='output dataset name (default: the dataset name)')
    ap.add_argument('--seed', type=int, default=3,
                    help='random state for split/subset selection (paper: 3)')
    ap.add_argument('--debug', action='store_true',
                    help='generate a small subset (where supported)')
    args = ap.parse_args(argv)

    fn = generators[args.dataset]
    params = inspect.signature(fn).parameters
    kwargs = {}
    if 'random_state' in params:
        kwargs['random_state'] = args.seed
    if 'debug' in params:
        kwargs['debug'] = args.debug
    elif args.debug:
        print(f'note: {args.dataset} has no --debug mode; generating the full dataset')
    if args.name:
        kwargs['dataset_name'] = args.name

    os.makedirs(args.out, exist_ok=True)
    fn(os.path.abspath(args.raw), os.path.abspath(args.out), **kwargs)
