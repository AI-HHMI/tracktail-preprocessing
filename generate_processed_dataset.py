
import os 

import numpy as np 
import pandas as pd 

from tracktail_preprocessing import cli
from tracktail_preprocessing.cli import KEYPOINTS_DIR
from tracktail_preprocessing.datasets import (
    ZefDataset,
    AcinosetDataset,
    AniposeFlyDataset,
    CMUPanopticDataset,
    CMUPanopticGSDataset,
    DexYCBDataset,
    KubricMultiviewDataset,
    PointOdysseyMultiviewDataset,
    PairR24MDataset,
    POPDataset,
    Rat7MDataset,
    TapVid3DDataset
)


def update_subsampling(splits, n_vids = 2, n_frames = 16): 

    split_dict = {}
    split_frames_dict = {}
    splits = set([split for split in splits if split != 'test'])

    for split in splits: 
        split_dict[split] = n_vids
        split_frames_dict[split] = n_frames

    return splits, split_dict, split_frames_dict


def generate_3dzef(dataset_path, out_prefix, dataset_name = '3dzef', 
                   random_state = 3, debug = False): 
    '''
    generates the preprocessed 3dzef dataset

    train: 8088 frames (all)
    val: None 
    test: 2710 frames (all)
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = ZefDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    # sample 8k training frames (full train dataset), generate full test 
    splits = {'train', 'test'}
    split_dict = {'train': None}
    split_frames_dict = {'train': None}

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict,
        random_state = random_state)

    # no validation data for this dataset
    dataset.generate_dataset(splits = splits)


def generate_acinoset(dataset_path, out_prefix, 
                      dataset_name = 'acinoset', 
                      random_state = 3, debug = False):
    
    '''
    generates the preprocessed acinoset dataset

    train: 20540 frames (all)
    val: 2 videos * 16  frames = 32 frames 
    test: 932 frames
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)
    keypoints_path = os.path.join(KEYPOINTS_DIR, f'keypoints_{dataset_name}.yaml')

    dataset = AcinosetDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name,
        keypoints_path = keypoints_path,
        filter_kernel_size = 11, 
        filter_thresh = None, 
        filter_percentile = 95)

    df = dataset.generate_metadata()

    # generate full training dataset (21k), full test data
    splits = {'train', 'val', 'test'}
    split_dict = {'train': None, 'val': 2} # number of videos to sample from the dataset
    split_frames_dict = {'train': None, 'val': 16} # number of consecutive frames per video to sample 

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)


def generate_anipose_fly(dataset_path, out_prefix, 
                         dataset_name = 'anipose_fly', 
                         random_state = 3, debug = False):

    ''' 
    generates the preprocessed anipose fly dataset
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = AniposeFlyDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    # sample 60k training frames, full training dataset
    splits = {'train', 'val', 'test'}
    split_dict = {'train': 3, 'val': 2} # number of videos to sample from the dataset
    split_frames_dict = {'train': 16, 'val': 16} # number of consecutive frames per video to sample 

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)


def generate_cmupanoptic(dataset_path, out_prefix, 
                         dataset_name = 'cmupanoptic', 
                         random_state = 3, debug = False): 

    ''' 
    generates the preprocessed cmupanoptic dataset
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = CMUPanopticDataset(
        dataset_path = dataset_path,
        dataset_outpath = dataset_outpath,
        dataset_name = dataset_name,
        keypoints_path = KEYPOINTS_DIR,
        conf_thresh = 0.1,
        filter_kernel_size = 11,
        filter_thresh = None,
        filter_percentile = 90)

    df = dataset.generate_metadata()

    # train: 57 sessions * 1000 frames = 57000 frames
    # val: 1 session * 32 frames = 32 frames
    splits = {'train', 'val', 'test'}
    split_dict = {'train': 57, 'val': 1} # number of videos to sample from the dataset
    split_frames_dict = {'train': 1000, 'val': 32} # number of consecutive frames per video to sample

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)


def generate_cmupanoptic3dgs(dataset_path, out_prefix, 
                             dataset_name = 'cmupanoptic_3dgs', 
                             random_state = 3): 
    ''' 
    generates the preprocessed cmupanoptic 3dgs dataset 

    train: None
    val: None
    test: 6 videos * 150 frames = 900 frames
    '''
    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = CMUPanopticGSDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    splits = {'test'}

    df = dataset.select_splits(
        random_state = random_state)

    # generate train and test splits
    dataset.generate_dataset(splits = splits)


def generate_dex_ycb(dataset_path, out_prefix, 
                     dataset_name = 'dex_ycb', 
                     random_state = 3): 
    ''' 
    generates the preprocessed dex ycb dataset

    train: None
    val: None
    test: 10 videos * ~24 frames = 245 frames
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = DexYCBDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    splits = {'test'}

    df = dataset.select_splits(
        random_state = random_state)
    
    # generate full dataset for testing 
    dataset.generate_dataset(splits = splits)


def generate_kubric_multiview(dataset_path, out_prefix, 
                              dataset_name = 'kubric-multiview', 
                              random_state = 3, debug = False):
    ''' 
    generates the preprocessed kubric multiview dataset

    train: 5000 videos * 24 frames = 120000 frames
    val: 2 videos * 24 frames = 48 frames
    test: 30 videos * 24 frames = 720 frames
    '''

    # generate full kubric multiview dataset for pretraining
    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = KubricMultiviewDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    splits = {'train', 'val', 'test'}
    split_dict = {'val': 2}
    split_frames_dict = {'val': 24}

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)
    
    df = dataset.select_splits(
        split_dict = split_dict,
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)


def generate_point_odyssey(dataset_path, out_prefix, dataset_name = 'point-odyssey-human',
                           random_state = 3, debug = False):
    '''
    generates the preprocessed point odyssey multiview dataset

    train: many sessions * T frames
    val: 1 session (flat split, no subdirectory)
    test: many sessions * T frames
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = PointOdysseyMultiviewDataset(
        dataset_path = dataset_path,
        dataset_outpath = dataset_outpath,
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    splits = {'train', 'val', 'test'}
    split_dict = {'val': 1}
    split_frames_dict = {'val': 32}

    if debug:
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict,
        split_frames_dict = split_frames_dict,
        random_state = random_state)

    dataset.generate_dataset(splits = splits)

def generate_point_odyssey_animal(dataset_path, out_prefix, dataset_name = 'point-odyssey-animal',
                                  random_state = 3, debug = False):
    '''
    generates the preprocessed point odyssey multiview animal dataset

    train: many sessions * T frames
    val: 1 session (flat split, no subdirectory)
    test: many sessions * T frames
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = PointOdysseyMultiviewDataset(
        dataset_path = dataset_path,
        dataset_outpath = dataset_outpath,
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    splits = {'train', 'val', 'test'}
    split_dict = {'val': 1}
    split_frames_dict = {'val': 32}

    if debug:
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict,
        split_frames_dict = split_frames_dict,
        random_state = random_state)

    dataset.generate_dataset(splits = splits)
    

def generate_pairr24m(dataset_path, out_prefix, dataset_name = 'pair-r24m',
                      random_state = 3, debug = False):
    '''
    generates the preprocessed pairr24m dataset

    train: 1039 videos * 49 frames = 50911 frames
    val: 2 videos * 16 frames = 32 frames
    test: 215910 frames
    '''
    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = PairR24MDataset(
        dataset_path = dataset_path,
        dataset_outpath = dataset_outpath,
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    # use all available train trials (1039 across 33 sessions)
    splits = {'train', 'val', 'test'}
    split_dict = {'train': 1039, 'val': 2} # number of videos to sample from the dataset
    split_frames_dict = {'train': 49, 'val': 16} # number of consecutive frames per video to sample

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)



def generate_3dpop(dataset_path, out_prefix, dataset_name = '3dpop', 
                   random_state = 3, debug = False): 
    ''' 
    generates the preprocessed 3dpop dataset

    train: 59 videos * 1017 frames = 60003 frames
    val: 2 videos * 16 frames = 32 frames
    test: 62901 frames
    '''
    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = POPDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name)

    df = dataset.generate_metadata()

    # sample 60k training frames
    splits = {'train', 'val', 'test'}
    split_dict = {'train': 59, 'val': 2} # number of videos to sample from the dataset
    split_frames_dict = {'train': 1017, 'val': 16} # number of consecutive frames per video to sample 

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)


def generate_rat7m(dataset_path, out_prefix, dataset_name = 'rat7m', 
                   random_state = 3, debug = False): 

    ''' 
    generates the preprocessed rat7m dataset 

    train: 190 videos * 320 frames = 60800 frames 
    val: 2 videos * 16 frames = 32 frames 
    test: 130 videos * 3500 frames = 455000 frames
    '''

    print(f'\ngenerating {dataset_name}...')
    dataset_outpath = os.path.join(out_prefix, dataset_name)

    dataset = Rat7MDataset(
        dataset_path = dataset_path, 
        dataset_outpath = dataset_outpath, 
        dataset_name = dataset_name,
        filter_kernel_size = 11, 
        filter_thresh = None, 
        filter_percentile = 90)

    df = dataset.generate_metadata()

    # sample 60k training frames, generate full test set
    splits = {'train', 'val', 'test'}
    split_dict = {'train': 190, 'val': 2} # number of videos to sample from the dataset
    split_frames_dict = {'train': 320, 'val': 16} # number of consecutive frames per video to sample 

    if debug: 
        splits, split_dict, split_frames_dict = update_subsampling(splits)

    df = dataset.select_splits(
        split_dict = split_dict, 
        split_frames_dict = split_frames_dict, 
        random_state = random_state)

    dataset.generate_dataset(splits = splits)


def _generate_tapvid3d(dataset_path, out_prefix, dataset_name):
    print(f'\ngenerating {dataset_name}...')
    dataset = TapVid3DDataset(
        dataset_path = dataset_path,
        dataset_outpath = os.path.join(out_prefix, dataset_name),
        dataset_name = dataset_name)
    dataset.generate_metadata()
    dataset.select_splits()
    dataset.generate_dataset(splits = {'test'})


def generate_tapvid3d_adt(dataset_path, out_prefix, dataset_name = 'tapvid3d_adt'):
    '''TAPVid-3D ADT subset (folder of official per-clip .npz files), test only'''
    _generate_tapvid3d(dataset_path, out_prefix, dataset_name)


def generate_tapvid3d_pstudio(dataset_path, out_prefix, dataset_name = 'tapvid3d_pstudio'):
    '''TAPVid-3D Panoptic Studio subset (folder of official per-clip .npz files), test only'''
    _generate_tapvid3d(dataset_path, out_prefix, dataset_name)


def generate_tapvid3d_drivetrack(dataset_path, out_prefix, dataset_name = 'tapvid3d_drivetrack'):
    '''TAPVid-3D DriveTrack subset (folder of official per-clip .npz files), test only'''
    _generate_tapvid3d(dataset_path, out_prefix, dataset_name)


if __name__ == '__main__':
    cli.main(cli.collect_generators(globals()),
             description='Convert a raw dataset into the Tracktail trial layout '
                         '(pretraining, paper evaluation and baseline-comparison datasets).')
