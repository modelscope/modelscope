# The implementation here is modified based on BaSSL,
# originally Apache 2.0 License and publicly available at https://github.com/kakaobrain/bassl
import copy
import os
import os.path as osp
import random

import json
import torch
from torchvision.datasets.folder import pil_loader

from modelscope.metainfo import Models
from modelscope.msdatasets.dataset_cls.custom_datasets.builder import \
    CUSTOM_DATASETS
from modelscope.utils.constant import Tasks
from . import sampler

DATASET_STRUCTURE = {
    'train': {
        'annotation': 'anno/train.json',
        'images': 'keyf_240p',
        'feat': 'feat'
    },
    'test': {
        'annotation': 'anno/test.json',
        'images': 'keyf_240p',
        'feat': 'feat'
    }
}


@CUSTOM_DATASETS.register_module(
    group_key=Tasks.movie_scene_segmentation, module_name=Models.resnet50_bert)
class MovieSceneSegmentationDataset(torch.utils.data.Dataset):
    """dataset for movie scene segmentation.

    Args:
        split_config (dict): Annotation file path. {"train":"xxxxx"}
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
    """

    def __init__(self, **kwargs):
        split_config = kwargs['split_config']

        self.data_root = next(iter(split_config.values()))
        if not osp.exists(self.data_root):
            self.data_root = osp.dirname(self.data_root)
            assert osp.exists(self.data_root)

        self.split = next(iter(split_config.keys()))
        self.preprocessor = kwargs['preprocessor']

        self.ann_file = osp.join(self.data_root,
                                 DATASET_STRUCTURE[self.split]['annotation'])
        self.img_prefix = osp.join(self.data_root,
                                   DATASET_STRUCTURE[self.split]['images'])
        self.feat_prefix = osp.join(self.data_root,
                                    DATASET_STRUCTURE[self.split]['feat'])

        self.test_mode = kwargs['test_mode']
        if self.test_mode:
            self.preprocessor.eval()
        else:
            self.preprocessor.train()

        self.cfg = kwargs.pop('cfg', None)

        self.num_keyframe = self.cfg.num_keyframe if self.cfg is not None else 3
        self.use_single_keyframe = self.cfg.use_single_keyframe if self.cfg is not None else False

        self.load_data()
        self.init_sampler(self.cfg)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.anno_data)

    def __getitem__(self, idx: int):
        data = self.anno_data[
            idx]  # {"video_id", "shot_id", "num_shot", "boundary_label"}
        vid, sid = data['video_id'], data['shot_id']
        num_shot = data['num_shot']

        shot_idx = self.shot_sampler(int(sid), num_shot)

        video = self.load_shot_list(vid, shot_idx)
        if self.preprocessor is None:
            video = torch.stack(video, dim=0)
            video = video.view(-1, self.num_keyframe, 3, 224, 224)
        else:
            video = self.preprocessor(video)

        payload = {
            'idx': idx,
            'vid': vid,
            'sid': sid,
            'video': video,
            'label': abs(data['boundary_label']),  # ignore -1 label.
        }
        return payload

    def load_data(self):
        self.tmpl = '{}/shot_{}_img_{}.jpg'  # video_id, shot_id, shot_num

        if not self.test_mode:
            with open(self.ann_file, encoding='utf-8') as f:
                self.anno_data = json.load(f)
            self.vidsid2label = {
                f"{it['video_id']}_{it['shot_id']}": it['boundary_label']
                for it in self.anno_data
            }
        else:
            with open(self.ann_file, encoding='utf-8') as f:
                self.anno_data = json.load(f)

    def init_sampler(self, cfg):
        # shot sampler
        if cfg is not None:
            self.sampling_method = cfg.sampling_method.name
            sampler_args = copy.deepcopy(
                cfg.sampling_method.params.get(self.sampling_method, {}))
            if self.sampling_method == 'instance':
                self.shot_sampler = sampler.InstanceShotSampler()
            elif self.sampling_method == 'temporal':
                self.shot_sampler = sampler.TemporalShotSampler(**sampler_args)
            elif self.sampling_method == 'shotcol':
                self.shot_sampler = sampler.SequenceShotSampler(**sampler_args)
            elif self.sampling_method == 'bassl':
                self.shot_sampler = sampler.SequenceShotSampler(**sampler_args)
            elif self.sampling_method == 'bassl+shotcol':
                self.shot_sampler = sampler.SequenceShotSampler(**sampler_args)
            elif self.sampling_method == 'sbd':
                self.shot_sampler = sampler.NeighborShotSampler(**sampler_args)
            else:
                raise NotImplementedError
        else:
            self.shot_sampler = sampler.NeighborShotSampler()

    def load_shot_list(self, vid, shot_idx):
        shot_list = []
        cache = {}
        for sidx in shot_idx:
            vidsid = f'{vid}_{sidx:04d}'
            if vidsid in cache:
                shot = cache[vidsid]
            else:
                shot_path = os.path.join(
                    self.img_prefix, self.tmpl.format(vid, f'{sidx:04d}',
                                                      '{}'))
                shot = self.load_shot_keyframes(shot_path)
                cache[vidsid] = shot
            shot_list.extend(shot)
        return shot_list

    def load_shot_keyframes(self, path):
        shot = None
        if not self.test_mode and self.use_single_keyframe:
            # load one randomly sampled keyframe
            shot = [
                pil_loader(
                    path.format(random.randint(0, self.num_keyframe - 1)))
            ]
        else:
            # load all keyframes
            shot = [
                pil_loader(path.format(i)) for i in range(self.num_keyframe)
            ]
        assert shot is not None
        return shot
