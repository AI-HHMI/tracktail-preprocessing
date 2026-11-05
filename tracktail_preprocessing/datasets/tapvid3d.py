"""TAPVid-3D (single camera, moving) converter.

Input: a folder of per-clip ``.npz`` files for one TAPVid-3D subset (``adt``, ``pstudio`` or
``drivetrack``), as written by the official generation scripts of
https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d. Each file has
``images_jpeg_bytes`` (T,), ``tracks_XYZ`` (T, N, 3) in camera coordinates, ``visibility``
(T, N), ``queries_xyt`` (N, 3) and ``fx_fy_cx_cy`` (4,).

Output (test split only), one trial per clip file ``<session>_<clip>.npz``::

    <out>/test/<session>/<clip>/
        metadata.yaml   one camera ``cam0``: K from fx_fy_cx_cy, identity extrinsics,
                        zero distortion, image size, num_frames, moving_cams: false
        pose3d.npz      pose (1, T, N, 3) float32, NaN where not visible
                        vis (1, T, N, 1) bool, keypoints ``kpt<i>``,
                        queries_xyt (N, 3) float32 (official query pixel x, y and frame)
        img/cam0/000000.jpg ...   the clip's JPEG bytes, written unchanged

Camera coordinates are used as world coordinates (extrinsics = identity), so the camera
motion of the clip is folded into the tracks, as in the benchmark.
"""

import glob
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from tracktail_preprocessing.datasets import BaseDataset
from tracktail_preprocessing.utils import io


class TapVid3DDataset(BaseDataset):

    def __init__(self, dataset_path, dataset_outpath, dataset_name='tapvid3d'):
        super().__init__(dataset_path, dataset_outpath)
        self.dataset_name = dataset_name

    @staticmethod
    def _split_name(path):
        # '<session>_<clip>.npz'; the clip id never contains '_'
        stem = os.path.splitext(os.path.basename(path))[0]
        session, clip = stem.rsplit('_', 1)
        return session, clip

    def load_calibration(self, clip, image_shape):
        fx, fy, cx, cy = [float(v) for v in clip['fx_fy_cx_cy']]
        h, w = image_shape[:2]
        cam = 'cam0'
        return {
            'intrinsic_matrices': {cam: [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]},
            'extrinsic_matrices': {cam: np.eye(4).tolist()},
            'distortion_matrices': {cam: np.zeros(5).tolist()},
            'camera_heights': {cam: int(h)},
            'camera_widths': {cam: int(w)},
        }

    def load_pose3d(self, clip):
        xyz = np.asarray(clip['tracks_XYZ'], dtype=np.float32).copy()   # (T, N, 3)
        vis = np.asarray(clip['visibility'], dtype=bool)                  # (T, N)
        xyz[~vis] = np.nan
        n = xyz.shape[1]
        return {
            'pose': xyz[None],
            'vis': vis[None, :, :, None],
            'keypoints': np.array([f'kpt{i}' for i in range(n)]),
            'queries_xyt': np.asarray(clip['queries_xyt'], dtype=np.float32),
        }

    def generate_metadata(self):
        rows = []
        for path in sorted(glob.glob(os.path.join(self.dataset_path, '*.npz'))):
            session, clip = self._split_name(path)
            rows.append({'id': f'{session}_{clip}', 'session': session, 'trial': clip,
                         'path': path, 'split': 'test', 'include': True})
        self.metadata = pd.DataFrame(rows)
        os.makedirs('metadata', exist_ok=True)
        self.metadata.drop(columns='path').to_csv(
            os.path.join('metadata', f'metadata_{self.dataset_name}.csv'), index=False)
        return self.metadata

    def select_splits(self, random_state=3):
        # TAPVid-3D is evaluation-only: every clip goes to test
        return self.metadata

    def generate_dataset(self, splits=None):
        df = self.metadata[self.metadata['include']]
        for _, row in tqdm(df.iterrows(), total=len(df), desc=self.dataset_name):
            outpath = os.path.join(self.dataset_outpath, 'test', row['session'], row['trial'])
            if os.path.exists(os.path.join(outpath, 'pose3d.npz')):
                continue
            with np.load(row['path'], allow_pickle=False) as clip:
                frames = clip['images_jpeg_bytes']
                img_dir = os.path.join(outpath, 'img', 'cam0')
                os.makedirs(img_dir, exist_ok=True)
                for i, jpg in enumerate(frames):
                    with open(os.path.join(img_dir, f'{i:06d}.jpg'), 'wb') as f:
                        f.write(bytes(jpg))
                first = cv2.imdecode(np.frombuffer(bytes(frames[0]), np.uint8), cv2.IMREAD_COLOR)
                meta = self.load_calibration(clip, first.shape)
                meta['moving_cams'] = False
                meta['num_cameras'] = 1
                meta['num_frames'] = int(len(frames))
                pose = self.load_pose3d(clip)
            io.save_yaml(data=meta, outpath=outpath, fname='metadata.yaml')
            np.savez_compressed(os.path.join(outpath, 'pose3d.npz'), **pose)
