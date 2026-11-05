#!/usr/bin/env python3
"""Cut each training trial of a converted Point Odyssey dataset to one fixed-length window.

This produced ``po-animal-sub256-motion`` (Tracktail paper, MV-PO animal) from the converted
``point-odyssey-animal`` dataset (``generate_processed_dataset.py --dataset point-odyssey-animal``)::

    python -m tracktail_preprocessing.scripts.make_subtime \
        --src OUT/point-odyssey-animal --dst OUT/po-animal-sub256-motion --frames 256 --mode motion

What it does, per split:

* ``train``: for every trial, positions that are not visible in any camera are set to NaN
  (``pose[~vis.any(-1)] = nan``), and one ``--frames``-long window is kept:

  - ``--mode motion`` (paper): the window start ``s`` maximizes
    ``sum_{t=s}^{s+F-2} sum_n ||X[t+1, n] - X[t, n]||`` over visible points (NaN terms skipped),
    i.e. the window with the most total 3D motion. Exhaustive over every start, ties -> earliest.
  - ``--mode first``: the first ``--frames`` frames.

  Output ``pose3d.npz`` holds float32 ``pose`` (1, F, N, 3), bool ``vis`` (1, F, N, C) and
  ``keypoints``. Images are symlinked and keep their source frame names
  (``img/<view>/img{start + i:06d}.jpg``). ``metadata.yaml`` gets ``num_frames`` and the
  provenance keys ``subtime_mode``, ``subtime_start`` and ``subtime_source_num_frames``.
* ``val`` and ``test``: the trial folders are symlinked unchanged (test stays at full length).

Trials shorter than ``--frames`` are kept whole. Resume-safe: trials whose output
``pose3d.npz`` already exists are skipped.

Provenance: the published ``po-animal-sub256-motion`` was made in July 2026 by an earlier version
of this script that is not preserved. This reimplementation reproduces its ``subtime_start`` and
pose/visibility arrays exactly on all trials checked (see the README). Run the drift cleaning
(``clean_pointodyssey_drift``) on the output afterwards.
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import yaml

SPLITS = ('train', 'val', 'test')


def save_npz_atomic(dst, arrays):
    d = os.path.dirname(dst)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.npz.tmp')
    try:
        with os.fdopen(fd, 'wb') as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def motion_window_start(pose, vis_any, n_frames):
    """pose (T, N, 3) with NaN for invisible points; returns the max-motion window start."""
    T = pose.shape[0]
    if T <= n_frames:
        return 0
    step = np.linalg.norm(np.diff(pose, axis=0), axis=-1)        # (T-1, N)
    per_t = np.nansum(step, axis=1)                                # (T-1,)
    csum = np.concatenate([[0.0], np.cumsum(per_t)])
    starts = np.arange(T - n_frames + 1)
    score = csum[starts + n_frames - 1] - csum[starts]             # n_frames-1 transitions
    return int(np.argmax(score))


def window_trial(src_trial, dst_trial, n_frames, mode):
    """src_trial / dst_trial are the inner ``trial/`` dirs."""
    dst_npz = os.path.join(dst_trial, 'pose3d.npz')
    if os.path.exists(dst_npz):
        return 'skipped'

    z = np.load(os.path.join(src_trial, 'pose3d.npz'))
    pose = z['pose'][0].astype(np.float64)                         # (T, N, 3)
    vis = z['vis'][0].astype(bool)                                 # (T, N, C)
    keypoints = z['keypoints']
    T = pose.shape[0]

    vis_any = vis.any(-1)
    pose[~vis_any] = np.nan

    n = min(n_frames, T)
    if mode == 'motion':
        start = motion_window_start(pose, vis_any, n)
    elif mode == 'first':
        start = 0
    else:
        raise ValueError(mode)
    sl = slice(start, start + n)

    # images: symlink the selected source frames, keeping their names
    src_img = os.path.join(src_trial, 'img')
    for view in sorted(os.listdir(src_img)):
        sv = os.path.join(src_img, view)
        if not os.path.isdir(sv):
            continue
        dv = os.path.join(dst_trial, 'img', view)
        os.makedirs(dv, exist_ok=True)
        frames = sorted(f for f in os.listdir(sv) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        for f in frames[sl]:
            dst = os.path.join(dv, f)
            if not os.path.lexists(dst):
                os.symlink(os.path.realpath(os.path.join(sv, f)), dst)

    with open(os.path.join(src_trial, 'metadata.yaml')) as fh:
        meta = yaml.safe_load(fh)
    meta['num_frames'] = n
    meta['subtime_mode'] = mode
    meta['subtime_source_num_frames'] = T
    meta['subtime_start'] = start
    with open(os.path.join(dst_trial, 'metadata.yaml'), 'w') as fh:
        yaml.dump(meta, fh)

    save_npz_atomic(dst_npz, dict(pose=pose[sl][None].astype(np.float32),
                                  vis=vis[sl][None], keypoints=keypoints))
    return f'{T}->{n} start={start}'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='converted dataset dir (with train/val/test)')
    ap.add_argument('--dst', required=True, help='output dataset dir')
    ap.add_argument('--frames', type=int, default=256, help='window length (paper: 256)')
    ap.add_argument('--mode', choices=('motion', 'first'), default='motion')
    ap.add_argument('--limit-trials', default=None,
                    help='comma-separated train trial names to restrict to')
    args = ap.parse_args(argv)

    limit = set(args.limit_trials.split(',')) if args.limit_trials else None
    n_err = 0
    for split in SPLITS:
        split_dir = os.path.join(args.src, split)
        if not os.path.isdir(split_dir):
            continue
        trials = sorted(t for t in os.listdir(split_dir)
                        if os.path.isdir(os.path.join(split_dir, t, 'trial')))
        if split != 'train':
            for t in trials:
                dst = os.path.join(args.dst, split, t, 'trial')
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.lexists(dst):
                    os.symlink(os.path.realpath(os.path.join(split_dir, t, 'trial')), dst)
            print(f'{split}: linked {len(trials)} trials')
            continue
        start_time = time.time()
        for i, t in enumerate(trials, 1):
            if limit is not None and t not in limit:
                continue
            try:
                status = window_trial(os.path.join(split_dir, t, 'trial'),
                                      os.path.join(args.dst, split, t, 'trial'),
                                      args.frames, args.mode)
            except Exception as e:  # keep going, report at the end
                status = f'ERROR {e}'
                n_err += 1
            print(f'[{i}/{len(trials)}] {time.time() - start_time:7.0f}s {split}/{t} {status}',
                  flush=True)
    return 1 if n_err else 0


if __name__ == '__main__':
    sys.exit(main())
