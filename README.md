# tracktail-preprocessing

Converters that turn public multi-view 3D tracking datasets into the trial layout read by
[Tracktail](https://github.com/AI-HHMI/tracktail).

## Install

```bash
git clone https://github.com/AI-HHMI/tracktail-preprocessing
cd tracktail-preprocessing
pixi install            # or: pip install -e .
```

## Output layout

Every converter writes

```
OUT_ROOT/<dataset>/<split>/<session>/<trial>/
    metadata.yaml   per-camera intrinsic_matrices, extrinsic_matrices (world-to-camera),
                    distortion_matrices, camera_heights, camera_widths; num_frames, num_cameras
    pose3d.npz      pose (S, T, N, 3) float, vis (S, T, N, C) bool, keypoints (N,)
    img/<camera>/<frame>.jpg
```

`<trial>` is `trial` for most datasets (TAPVid-3D uses the clip id). A CSV of the selected
trials and splits is written to `./metadata/metadata_<dataset>.csv`; the ones used for our
data are committed in `metadata/`.

## Converting a dataset

```bash
pixi run python generate_processed_dataset.py        --dataset NAME --in RAW_DIR --out OUT_ROOT [--seed 3] [--debug]
pixi run python generate_processed_dataset_animal.py --dataset NAME --in RAW_DIR --out OUT_ROOT [--seed 3] [--debug]
```

`--in` is the raw dataset folder itself; output goes to `OUT_ROOT/NAME/`. `--seed 3` (the
default) is the seed used for all our data. `--help` lists the datasets.
`generate_processed_dataset.py` has the pretraining / evaluation datasets used in the paper,
`generate_processed_dataset_animal.py` the animal finetuning datasets.

## Datasets used in the Tracktail paper

The paper's data root (`TRACKTAIL_DATA` in the tracktail repo) holds these folders. Below, `RAW` is
where you downloaded the raw data and `DATA` is the data root.

> TODO: download links for Multi-view Kubric and Multi-view Point Odyssey (human/animal) — upload in progress.
> These are released already processed (`kubric-multiview`, `point-odyssey-human`,
> `po-animal-sub256-motion`); you don't need to run anything below for them.

Files other than `metadata.yaml`, `pose3d.npz` and `img/` that you may see in our trial folders
(`duster-views-*`, `gt-depths`, `pose3d_cached*.npz`, `pose3d_ncam*_512.npz`, ...) are
baseline caches and track subsets made by tracktail repo scripts, not by this repo.

### kubric-multiview (MV-Kubric)

- Processed data: HF (TODO link). Splits: train 4989, val 2, test 30 scenes, 24 frames each.
- Raw renders come from the MVTracker Kubric fork
  ([ethz-vlg/kubric@multiview-point-tracking](https://github.com/ethz-vlg/kubric/blob/multiview-point-tracking/challenges/point_tracking_3d/worker.py));
  the 30 test scenes are also in `kubric-multiview--test.tar.gz` of
  [ethz-vlg/mv3dpt-datasets](https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets).
- Convert raw per-camera renders (`RAW/kubric-multiview/{train,val,test}/<scene>/<camera>/`, each with the
  worker's `metadata.json`; not the MVTracker test tarball format):
  ```bash
  pixi run python generate_processed_dataset.py --dataset kubric-multiview --in RAW/kubric-multiview --out DATA
  ```
- Output: `DATA/kubric-multiview/{train,val,test}/<scene>/trial/`

### dex_ycb (DexYCB, 8 views)

- Raw: `dex-ycb-multiview.tar.gz` from [ethz-vlg/mv3dpt-datasets](https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets)
  (MVTracker's labelled 10 sequences; see their README for the DexYCB license).
  ```bash
  wget https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets/resolve/main/dex-ycb-multiview.tar.gz -P RAW
  tar -xzf RAW/dex-ycb-multiview.tar.gz -C RAW
  pixi run python generate_processed_dataset.py --dataset dex_ycb --in RAW/dex-ycb-multiview --out DATA
  ```
- Output: `DATA/dex_ycb/test/<subject>__<sequence>/trial/` (10 trials, test only)

### cmupanoptic_3dgs (Panoptic Studio, Dynamic3DGS tracks)

- Raw: `panoptic-multiview.tar.gz` from [ethz-vlg/mv3dpt-datasets](https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets).
  ```bash
  wget https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets/resolve/main/panoptic-multiview.tar.gz -P RAW
  tar -xzf RAW/panoptic-multiview.tar.gz -C RAW
  pixi run python generate_processed_dataset.py --dataset cmupanoptic_3dgs --in RAW/panoptic-multiview --out DATA
  ```
- Output: `DATA/cmupanoptic_3dgs/test/{basketball,boxes,football,juggle,softball,tennis}/trial/` (150 frames each)

### point-odyssey-human and po-animal-sub256-motion (MV-PO)

- Processed data: HF (TODO link). We don't ship the generation code for the raw renders (a
  modified [PointOdyssey](https://github.com/y-zheng18/point_odyssey) Blender pipeline with 9
  static cameras; available on request). The steps after rendering are all here, in this order:

  1. **Convert** the renders (`RAW/point-odyssey/results{,-animal}/{train,val,test}/<session>/view*/{annotations.npz,rgbs/,exr_img/}`).
     The converter computes per-camera visibility with a depth test against the rendered depth maps;
     images are symlinked, not copied.
     ```bash
     pixi run python generate_processed_dataset.py --dataset point-odyssey-human  --in RAW/point-odyssey/results        --out STAGE
     pixi run python generate_processed_dataset.py --dataset point-odyssey-animal --in RAW/point-odyssey/results-animal --out STAGE
     ```
  2. **Window (animal train only)**: keep the 256-frame window with the most 3D motion of
     visible points per train trial; val/test are linked unchanged (test stays 499 frames).
     ```bash
     pixi run python -m tracktail_preprocessing.scripts.make_subtime \
         --src STAGE/point-odyssey-animal --dst STAGE/po-animal-sub256-motion --frames 256 --mode motion
     ```
  3. **Clean tracks** (both datasets, all splits): drop a track for the whole clip if it ever moves
     >100 px/frame in any camera, comes within depth 1 of a camera, or strays >20x the scene's
     robust radius from its centre.
     ```bash
     pixi run python -m tracktail_preprocessing.scripts.clean_pointodyssey_drift \
         --src-prefix STAGE --dst-prefix DATA \
         --datasets point-odyssey-human po-animal-sub256-motion --workers 8
     ```
  4. **Test track subset** (in the tracktail repo, not here): `scripts/make_pose3d_512.py` replaces each
     test `pose3d.npz` by a 512-track speed-stratified subset and keeps the full cleaned file as
     `pose3d_full.npz`.

- Output: `DATA/point-odyssey-human/{train,val,test}/<hum*|val>/trial/` (113 / 1 / 8) and
  `DATA/po-animal-sub256-motion/{train,val,test}/<ani*|val>/trial/` (210 / 1 / 10).
- Notes: the paper's animal test set has 10 of the 12 converted test trials (`ani1` and `ani8`
  were removed from it). `make_subtime.py` is a reimplementation: the July 2026 script that made
  `po-animal-sub256-motion` was not kept. It picks the same window (`subtime_start`) as the
  released data on all 210 train trials, and its arrays, metadata and image links match exactly
  (checked on 3 trials end to end, through the cleaning step).

### tapvid3d_adt, tapvid3d_drivetrack, tapvid3d_pstudio (TAPVid-3D)

- Raw: generate the per-clip `.npz` files with the official scripts in
  [google-deepmind/tapnet/tapnet/tapvid3d](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d)
  (one folder per subset: `adt/`, `drivetrack/`, `pstudio/`, files named `<session>_<clip>.npz`).
  ```bash
  for s in adt drivetrack pstudio; do
      pixi run python generate_processed_dataset.py --dataset tapvid3d_$s --in RAW/tapvid3d/$s --out DATA
  done
  ```
- Output: `DATA/tapvid3d_<subset>/test/<session>/<clip>/` with one camera `cam0` (identity
  extrinsics, i.e. camera coordinates), `pose` NaN where not visible, and the benchmark's
  `queries_xyt` in `pose3d.npz`. All clips are converted; the paper's evaluation subset is
  selected in the tracktail repo (`arrow_baseline/tapvid3d_minival.json`).

### rat7m (Rat7M, cover figure)

- Raw: [Rat 7M](https://doi.org/10.6084/m9.figshare.c.5295370) (Dunn et al. 2021): motion
  capture `.mat` files, camera calibration and videos, arranged as `RAW/rat7m/{data,videos}/`.
  ```bash
  pixi run python generate_processed_dataset.py --dataset rat7m --in RAW/rat7m --out DATA
  ```
- Output: `DATA/rat7m/{train,val,test}/<session>/...` (train: 190 x 320-frame windows; test: all 3500-frame chunks).

## Other scripts

- `python -m tracktail_preprocessing.scripts.fix_metadata_camera_keys`, `migrate_metadata_keys`:
  one-off metadata migrations for older processed folders.
- `python -m tracktail_preprocessing.scripts.preview_bouts`: preview images for the anipose fly
  bout selection.

## License

BSD-3-Clause. Each dataset has its own license; check it before use.
