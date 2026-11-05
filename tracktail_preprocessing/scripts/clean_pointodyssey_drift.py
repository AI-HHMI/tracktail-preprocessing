#!/usr/bin/env python3
"""Clean drift / near-camera-projection artifacts out of converted Point Odyssey datasets.

This is the "track cleaning" step of MV-PO in the Tracktail paper. It produced the
``point-odyssey-human`` and ``po-animal-sub256-motion`` folders of the paper's data root
(``posetail-pretraining-v6``) on 2026-09-04 (output trials carry
``cleaning.script_version: clean_pointodyssey_drift.py v1`` in ``metadata.yaml``)::

    python -m tracktail_preprocessing.scripts.clean_pointodyssey_drift \
        --src-prefix OUT_UNCLEANED --dst-prefix OUT_CLEAN \
        --datasets point-odyssey-human po-animal-sub256-motion --workers 8

``--src-prefix`` holds ``<dataset>/<split>/<trial>/trial/{pose3d.npz,metadata.yaml,img}``.
Design notes: tracktail repo, ``dev/plans/clean-pointodyssey-drift-artifacts.md``.

Cleaning rule (whole-trajectory drop):

For each trial, each keypoint n is either KEPT for its entire trajectory or DROPPED for its
entire trajectory (every frame, every camera, the 3D coordinate itself set to NaN). A
keypoint is dropped if EITHER:

  (A/A2) at ANY consecutive-frame transition (t, t+1), in ANY camera, the projected pixel
         speed exceeds --speed-thresh-px, OR either endpoint's depth in that camera is
         <= --z-min (near-zero-depth / behind-camera degenerate projection); OR
  (B)    at ANY frame, the keypoint's 3D position is farther than --spatial-mult times a
         robust per-trial scale from a robust per-trial center, both computed from only the
         first K_REF frames (always clean in every trial inspected during development).

Everything else (images, camera calibration) is untouched; images are symlinked, never copied.
``--mirror-only`` symlinks every other dataset of ``--src-prefix`` into ``--dst-prefix``.

Stdlib + numpy + pyyaml only. Atomic writes (temp file + os.replace), resume-safe (skips any
trial whose output pose3d.npz + metadata.yaml + img symlink already exist).
"""

import argparse
import concurrent.futures
import os
import sys
import tempfile
import time

import numpy as np
import yaml

DATASETS = ('point-odyssey-human', 'po-animal-sub256-motion')
SPLITS = ('train', 'val', 'test')


# --------------------------------------------------------------------------- #
# atomic write helpers (matches make_subtime.py conventions)
# --------------------------------------------------------------------------- #
def write_bytes_atomic(dst, data):
    if os.path.exists(dst):
        return False
    d = os.path.dirname(dst)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(data)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return True


def save_npz_atomic(dst, arrays, compressed=True):
    if os.path.exists(dst):
        return False
    d = os.path.dirname(dst)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.npz.tmp')
    try:
        with os.fdopen(fd, 'wb') as fh:
            if compressed:
                np.savez_compressed(fh, **arrays)
            else:
                np.savez(fh, **arrays)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return True


def symlink_atomic(target, dst):
    """Create dst -> target. If dst already exists as a symlink, verify it resolves to the
    SAME real path as `target` (not just that it exists) -- a stale/incorrect pre-existing
    entry is a correctness bug (silently pointing the output at different data than the source), not a
    resume no-op, so it is corrected rather than skipped. Returns 'created', 'already-correct',
    or 'fixed-mismatch'.
    """
    real_target = os.path.realpath(target)
    if os.path.islink(dst):
        if os.path.realpath(dst) == real_target:
            return 'already-correct'
        print(f'  [FIX] {dst} pointed to {os.path.realpath(dst)}, expected {real_target} -- correcting')
        os.remove(dst)
    elif os.path.exists(dst):
        raise FileExistsError(f'{dst} exists and is not a symlink -- refusing to overwrite')
    d = os.path.dirname(dst)
    os.makedirs(d, exist_ok=True)
    tmp = dst + '.tmp-symlink'
    if os.path.islink(tmp):
        os.remove(tmp)
    os.symlink(real_target, tmp)
    os.replace(tmp, dst)
    return 'created'


# --------------------------------------------------------------------------- #
# camera geometry / projection (validated against dataset during investigation)
# --------------------------------------------------------------------------- #
def load_cam_geometry(meta):
    exts = meta['extrinsic_matrices']
    ints = meta['intrinsic_matrices']
    names = sorted(exts.keys())
    cams = {}
    for name in names:
        E = np.array(exts[name], dtype=np.float64)
        K = np.array(ints[name], dtype=np.float64)
        R = E[:3, :3]
        t = E[:3, 3]
        cams[name] = dict(
            R=R, t=t,
            f=float(K[0, 0]), cx=float(K[0, 2]), cy=float(K[1, 2]),
            w=int(meta['camera_widths'][name]), h=int(meta['camera_heights'][name]),
        )
    return names, cams


def project(P, cam):
    """P: (..., 3) world points -> (pixel (...,2), depth (...,)).

    z<=0 (behind/at camera) legitimately produces inf/nan here; that's fine -- the z<=Z_MIN
    check downstream flags those points independent of what the resulting pixel value is, so
    the divide warning is expected noise, not a bug.
    """
    X = P @ cam['R'].T + cam['t']
    z = X[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        x = X[..., 0] * cam['f'] / z + cam['cx']
        y = X[..., 1] * cam['f'] / z + cam['cy']
    return np.stack([x, y], axis=-1), z


# --------------------------------------------------------------------------- #
# core cleaning
# --------------------------------------------------------------------------- #
def clean_trial(pose, vis, cam_names, cams, speed_thresh, z_min, spatial_mult, ref_frames):
    """pose: (T,N,3) float64, vis: (T,N,C) bool. Returns (pose_out, vis_out, stats dict)."""
    T, N, _ = pose.shape
    K_REF = int(min(ref_frames, max(16, round(0.05 * T)), T))

    ref_block = pose[:K_REF]
    fin = np.isfinite(ref_block).all(-1)
    ref_pts = ref_block[fin]
    if ref_pts.shape[0] < 10:
        fin_all = np.isfinite(pose).all(-1)
        ref_pts = pose[fin_all]
    if ref_pts.shape[0] < 2:
        # degenerate trial: nothing finite to reference against. Keep everything (can't judge).
        stats = dict(N=N, n_dropped=0, n_spatial=0, n_speed_or_depth=0,
                     ref_scale=float('nan'), K_REF=K_REF, degenerate_ref=True)
        return pose.copy(), vis.copy(), stats

    ref_center = np.median(ref_pts, axis=0)
    ref_scale = float(np.median(np.linalg.norm(ref_pts - ref_center, axis=-1)))
    degenerate_ref = not np.isfinite(ref_scale) or ref_scale <= 0
    if degenerate_ref:
        ref_scale = 1.0

    drop_point = np.zeros(N, dtype=bool)

    dist = np.linalg.norm(pose - ref_center, axis=-1)   # (T, N)
    bad_spatial = np.isfinite(dist) & (dist > spatial_mult * ref_scale)
    n_spatial = int(bad_spatial.any(axis=0).sum())
    drop_point |= bad_spatial.any(axis=0)

    n_speed_or_depth = 0
    for name in cam_names:
        cam = cams[name]
        p, z = project(pose.reshape(-1, 3), cam)
        p = p.reshape(T, N, 2)
        z = z.reshape(T, N)
        # NaN endpoints propagate to NaN speed/z; NaN comparisons evaluate False, so a
        # transition with a missing endpoint is correctly never flagged here.
        speed = np.linalg.norm(np.diff(p, axis=0), axis=-1)   # (T-1, N)
        bad_cam = (speed > speed_thresh) | (z[:-1] <= z_min) | (z[1:] <= z_min)
        newly = bad_cam.any(axis=0) & ~drop_point
        n_speed_or_depth += int(newly.sum())
        drop_point |= bad_cam.any(axis=0)

    pose_out = pose.copy()
    pose_out[:, drop_point, :] = np.nan
    vis_out = vis.copy()
    vis_out[:, drop_point, :] = False

    stats = dict(N=N, n_dropped=int(drop_point.sum()), n_spatial=n_spatial,
                 n_speed_or_depth=n_speed_or_depth, ref_scale=ref_scale, K_REF=K_REF,
                 degenerate_ref=degenerate_ref)
    return pose_out, vis_out, stats


# --------------------------------------------------------------------------- #
# per-trial driver
# --------------------------------------------------------------------------- #
def process_trial(src_trial_dir, dst_trial_dir, args):
    """src_trial_dir / dst_trial_dir are the outer <trial> dirs (containing trial/)."""
    src_inner = os.path.join(src_trial_dir, 'trial')
    dst_inner = os.path.join(dst_trial_dir, 'trial')

    src_npz = os.path.join(src_inner, 'pose3d.npz')
    src_meta = os.path.join(src_inner, 'metadata.yaml')
    src_img = os.path.join(src_inner, 'img')

    dst_npz = os.path.join(dst_inner, 'pose3d.npz')
    dst_meta = os.path.join(dst_inner, 'metadata.yaml')
    dst_img = os.path.join(dst_inner, 'img')

    already_done = os.path.exists(dst_npz) and os.path.exists(dst_meta) and os.path.islink(dst_img)
    if already_done:
        return {'status': 'skipped-resume'}

    z = np.load(src_npz)
    pose_full = z['pose']
    vis_full = z['vis']
    keypoints = z['keypoints']
    assert pose_full.shape[0] == 1, f'expected S=1, got {pose_full.shape[0]} in {src_npz}'
    assert vis_full.shape[0] == 1, f'expected S=1, got {vis_full.shape[0]} in {src_npz}'

    pose = pose_full[0].astype(np.float64)
    vis = vis_full[0].astype(bool)

    meta = yaml.safe_load(open(src_meta))
    cam_names, cams = load_cam_geometry(meta)
    # vis camera axis order must match cam_names order used here; verify by count.
    assert vis.shape[-1] == len(cam_names), (
        f'{src_npz}: vis has {vis.shape[-1]} camera columns but metadata has '
        f'{len(cam_names)} cameras ({cam_names})')

    if args.dry_run:
        pose_out, vis_out, stats = clean_trial(
            pose, vis, cam_names, cams, args.speed_thresh_px, args.z_min,
            args.spatial_mult, args.spatial_ref_frames)
        stats['trial'] = os.path.basename(src_trial_dir)
        stats['T'] = int(pose.shape[0])
        return {'status': 'dry-run', 'stats': stats}

    pose_out, vis_out, stats = clean_trial(
        pose, vis, cam_names, cams, args.speed_thresh_px, args.z_min,
        args.spatial_mult, args.spatial_ref_frames)

    # restore the S=1 leading axis; write float32 + bool, compressed (bundled fix)
    pose_save = pose_out[None].astype(np.float32)
    vis_save = vis_out[None].astype(bool)
    save_npz_atomic(dst_npz, dict(pose=pose_save, vis=vis_save, keypoints=keypoints))

    meta_out = dict(meta)
    meta_out['cleaning'] = {
        'source': src_trial_dir,
        'drop_granularity': 'whole_trajectory',
        'speed_thresh_px_per_frame': args.speed_thresh_px,
        'near_zero_depth_min_z': args.z_min,
        'spatial_excursion_multiplier': args.spatial_mult,
        'spatial_reference_frames': stats['K_REF'],
        'n_keypoints_total': stats['N'],
        'n_keypoints_dropped': stats['n_dropped'],
        'n_dropped_by_spatial': stats['n_spatial'],
        'n_dropped_by_speed_or_depth': stats['n_speed_or_depth'],
        'ref_scale': stats['ref_scale'],
        'degenerate_ref': stats['degenerate_ref'],
        'script_version': SCRIPT_VERSION,
    }
    write_bytes_atomic(dst_meta, yaml.safe_dump(meta_out, sort_keys=False).encode())

    real_img = os.path.realpath(src_img)
    symlink_atomic(real_img, dst_img)

    stats['trial'] = os.path.basename(src_trial_dir)
    stats['T'] = int(pose.shape[0])
    return {'status': 'written', 'stats': stats}


SCRIPT_VERSION = 'clean_pointodyssey_drift.py v1'


# --------------------------------------------------------------------------- #
# mirroring untouched datasets
# --------------------------------------------------------------------------- #
def mirror_untouched(src_prefix, dst_prefix, datasets_in_scope):
    os.makedirs(dst_prefix, exist_ok=True)
    counts = {'created': 0, 'already-correct': 0, 'fixed-mismatch': 0}
    for entry in sorted(os.listdir(src_prefix)):
        if entry in datasets_in_scope:
            continue
        src = os.path.join(src_prefix, entry)
        dst = os.path.join(dst_prefix, entry)
        before = os.path.realpath(dst) if os.path.islink(dst) else None
        status = symlink_atomic(src, dst)
        if status == 'created' and before is not None:
            status = 'fixed-mismatch'
        counts[status] = counts.get(status, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# discovery + orchestration
# --------------------------------------------------------------------------- #
def find_trials(dataset_split_dir):
    if not os.path.isdir(dataset_split_dir):
        return []
    return sorted(os.listdir(dataset_split_dir))


def fmt_dur(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


def run_one(job, args):
    """Top-level (picklable) so this works under ProcessPoolExecutor, which pickles the
    submitted callable + args regardless of fork/spawn start method."""
    dataset, split, trial, src_trial_dir, dst_trial_dir = job
    try:
        r = process_trial(src_trial_dir, dst_trial_dir, args)
        r['dataset'] = dataset
        r['split'] = split
        r['trial'] = trial
        return r
    except Exception as e:
        return {'status': 'error', 'dataset': dataset, 'split': split, 'trial': trial,
                'error': str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src-prefix', required=True,
                    help='data root with the uncleaned <dataset>/<split>/<trial>/trial dirs')
    ap.add_argument('--dst-prefix', required=True, help='output data root')
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS))
    ap.add_argument('--splits', nargs='+', default=list(SPLITS))
    ap.add_argument('--speed-thresh-px', type=float, default=100.0)
    ap.add_argument('--z-min', type=float, default=1.0)
    ap.add_argument('--spatial-mult', type=float, default=20.0)
    ap.add_argument('--spatial-ref-frames', type=int, default=32)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--limit-trials', default=None,
                    help='comma-separated trial names to restrict processing to (dry-run subset)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--mirror-only', action='store_true')
    args = ap.parse_args()

    if args.mirror_only:
        counts = mirror_untouched(args.src_prefix, args.dst_prefix, set(DATASETS))
        print(f'mirror into {args.dst_prefix}: {counts}')
        return 0

    limit = set(args.limit_trials.split(',')) if args.limit_trials else None

    jobs = []
    for dataset in args.datasets:
        for split in args.splits:
            split_dir = os.path.join(args.src_prefix, dataset, split)
            trials = find_trials(split_dir)
            for trial in trials:
                if limit is not None and trial not in limit:
                    continue
                src_trial_dir = os.path.join(split_dir, trial)
                dst_trial_dir = os.path.join(args.dst_prefix, dataset, split, trial)
                jobs.append((dataset, split, trial, src_trial_dir, dst_trial_dir))

    print(f'{len(jobs)} trials queued '
          f'({"DRY RUN, no writes" if args.dry_run else "writing to " + args.dst_prefix})')

    results = []
    start = time.time()

    if args.workers <= 1:
        for i, job in enumerate(jobs, 1):
            r = run_one(job, args)
            results.append(r)
            elapsed = time.time() - start
            rate = elapsed / i
            eta = rate * (len(jobs) - i)
            print(f'[{i}/{len(jobs)} {100.0*i/max(len(jobs),1):4.1f}%] elapsed {fmt_dur(elapsed)} '
                  f'| {rate:.1f}s/trial | ETA {fmt_dur(eta)} | {r["dataset"]}/{r["split"]}/'
                  f'{r["trial"]} {r["status"]}', flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, job, args): job for job in jobs}
            i = 0
            for fut in concurrent.futures.as_completed(futs):
                i += 1
                r = fut.result()
                results.append(r)
                elapsed = time.time() - start
                rate = elapsed / i
                eta = rate * (len(jobs) - i)
                print(f'[{i}/{len(jobs)} {100.0*i/max(len(jobs),1):4.1f}%] elapsed '
                      f'{fmt_dur(elapsed)} | {rate:.1f}s/trial | ETA {fmt_dur(eta)} | '
                      f'{r["dataset"]}/{r["split"]}/{r["trial"]} {r["status"]}', flush=True)

    n_err = sum(1 for r in results if r['status'] == 'error')
    print(f'Done: {len(results)} trials, {n_err} errors, {fmt_dur(time.time()-start)}')

    if args.dry_run:
        print()
        print('=== per-dataset drop summary ===')
        for dataset in args.datasets:
            rows = [r for r in results if r['dataset'] == dataset and 'stats' in r]
            if not rows:
                continue
            tot_n = sum(r['stats']['N'] for r in rows)
            tot_drop = sum(r['stats']['n_dropped'] for r in rows)
            tot_spatial = sum(r['stats']['n_spatial'] for r in rows)
            tot_speed = sum(r['stats']['n_speed_or_depth'] for r in rows)
            print(f'{dataset}: {len(rows)} trials, {tot_n} total keypoints, '
                  f'{tot_drop} dropped ({100.0*tot_drop/max(tot_n,1):.2f}%) '
                  f'[spatial={tot_spatial} speed/depth={tot_speed}]')
            for r in rows:
                s = r['stats']
                print(f'    {r["trial"]:10s} T={s["T"]:5d} N={s["N"]:6d} '
                      f'dropped={s["n_dropped"]:6d} ({100.0*s["n_dropped"]/max(s["N"],1):5.1f}%) '
                      f'spatial={s["n_spatial"]:6d} speed/depth={s["n_speed_or_depth"]:6d} '
                      f'ref_scale={s["ref_scale"]:.2f} K_REF={s["K_REF"]}'
                      f'{" DEGENERATE_REF" if s.get("degenerate_ref") else ""}')

    return 1 if n_err else 0


if __name__ == '__main__':
    sys.exit(main())
