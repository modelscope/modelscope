# Part of the implementation is borrowed and modified from MTTR,
# publicly available at https://github.com/mttr2021/MTTR

from glob import glob
from os import path as osp

import h5py
import json
import numpy as np
import pandas
import torch
import torch.distributed as dist
import torchvision.transforms.functional as F
from pycocotools.mask import area, encode
from torchvision.io import read_video
from tqdm import tqdm

from modelscope.metainfo import Models
from modelscope.models.cv.referring_video_object_segmentation.utils import \
    nested_tensor_from_videos_list
from modelscope.msdatasets.task_datasets.builder import TASK_DATASETS
from modelscope.msdatasets.task_datasets.torch_base_dataset import \
    TorchTaskDataset
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from . import transformers as T

LOGGER = get_logger()


def get_image_id(video_id, frame_idx, ref_instance_a2d_id):
    image_id = f'v_{video_id}_f_{frame_idx}_i_{ref_instance_a2d_id}'
    return image_id


@TASK_DATASETS.register_module(
    Tasks.referring_video_object_segmentation,
    module_name=Models.referring_video_object_segmentation)
class ReferringVideoObjectSegmentationDataset(TorchTaskDataset):

    def __init__(self, **kwargs):
        split_config = kwargs['split_config']
        LOGGER.info(kwargs)
        data_cfg = kwargs.get('cfg').data_kwargs
        trans_cfg = kwargs.get('cfg').transformers_kwargs
        distributed = data_cfg.get('distributed', False)

        self.data_root = next(iter(split_config.values()))
        if not osp.exists(self.data_root):
            self.data_root = osp.dirname(self.data_root)
            assert osp.exists(self.data_root)

        self.window_size = data_cfg.get('window_size', 8)
        self.mask_annotations_dir = osp.join(
            self.data_root, 'text_annotations/annotation_with_instances')
        self.videos_dir = osp.join(self.data_root, 'Release/CLIPS320')
        self.subset_type = next(iter(split_config.keys()))
        self.text_annotations = self.get_text_annotations(
            self.data_root, self.subset_type, distributed)
        self.transforms = A2dSentencesTransforms(self.subset_type, **trans_cfg)
        self.collator = Collator()
        self.ann_file = osp.join(
            self.data_root,
            data_cfg.get('ann_file',
                         'a2d_sentences_test_annotations_in_coco_format.json'))

        # create ground-truth test annotations for the evaluation process if necessary:
        if self.subset_type == 'test' and not osp.exists(self.ann_file):
            if (distributed and dist.get_rank() == 0) or not distributed:
                create_a2d_sentences_ground_truth_test_annotations(
                    self.data_root, self.subset_type,
                    self.mask_annotations_dir, self.ann_file)
            if distributed:
                dist.barrier()

    def __len__(self):
        return len(self.text_annotations)

    def __getitem__(self, idx):
        text_query, video_id, frame_idx, instance_id = self.text_annotations[
            idx]

        text_query = ' '.join(
            text_query.lower().split())  # clean up the text query

        # read the source window frames:
        video_frames, _, _ = read_video(
            osp.join(self.videos_dir, f'{video_id}.mp4'),
            pts_unit='sec')  # (T, H, W, C)
        # get a window of window_size frames with frame frame_idx in the middle.
        # note that the original a2d dataset is 1 indexed, so we have to subtract 1 from frame_idx
        start_idx, end_idx = frame_idx - 1 - self.window_size // 2, frame_idx - 1 + (
            self.window_size + 1) // 2

        # extract the window source frames:
        source_frames = []
        for i in range(start_idx, end_idx):
            i = min(max(i, 0),
                    len(video_frames)
                    - 1)  # pad out of range indices with edge frames
            source_frames.append(
                F.to_pil_image(video_frames[i].permute(2, 0, 1)))

        # read the instance mask:
        frame_annot_path = osp.join(self.mask_annotations_dir, video_id,
                                    f'{frame_idx:05d}.h5')
        f = h5py.File(frame_annot_path, 'r')
        instances = list(f['instance'])
        instance_idx = instances.index(
            instance_id)  # existence was already validated during init

        instance_masks = np.array(f['reMask'])
        if len(instances) == 1:
            instance_masks = instance_masks[np.newaxis, ...]
        instance_masks = torch.tensor(instance_masks).transpose(1, 2)
        mask_rles = [encode(mask) for mask in instance_masks.numpy()]
        mask_areas = area(mask_rles).astype(np.float)
        f.close()

        # create the target dict for the center frame:
        target = {
            'masks': instance_masks,
            'orig_size': instance_masks.
            shape[-2:],  # original frame shape without any augmentations
            # size with augmentations, will be changed inside transforms if necessary
            'size': instance_masks.shape[-2:],
            'referred_instance_idx': torch.tensor(
                instance_idx),  # idx in 'masks' of the text referred instance
            'area': torch.tensor(mask_areas),
            'iscrowd':
            torch.zeros(len(instance_masks)
                        ),  # for compatibility with DETR COCO transforms
            'image_id': get_image_id(video_id, frame_idx, instance_id)
        }

        # create dummy targets for adjacent frames:
        targets = self.window_size * [None]
        center_frame_idx = self.window_size // 2
        targets[center_frame_idx] = target
        source_frames, targets, text_query = self.transforms(
            source_frames, targets, text_query)
        return source_frames, targets, text_query

    @staticmethod
    def get_text_annotations(root_path, subset, distributed):
        saved_annotations_file_path = osp.join(
            root_path, f'sentences_single_frame_{subset}_annotations.json')
        if osp.exists(saved_annotations_file_path):
            with open(saved_annotations_file_path, 'r', encoding='utf-8') as f:
                text_annotations_by_frame = [tuple(a) for a in json.load(f)]
                return text_annotations_by_frame
        elif (distributed and dist.get_rank() == 0) or not distributed:
            print(f'building a2d sentences {subset} text annotations...')
            # without 'header == None' pandas will ignore the first sample...
            a2d_data_info = pandas.read_csv(
                osp.join(root_path, 'Release/videoset.csv'), header=None)
            # 'vid', 'label', 'start_time', 'end_time', 'height', 'width', 'total_frames', 'annotated_frames', 'subset'
            a2d_data_info.columns = [
                'vid', '', '', '', '', '', '', '', 'subset'
            ]
            with open(
                    osp.join(root_path, 'text_annotations/missed_videos.txt'),
                    'r') as f:
                unused_videos = f.read().splitlines()
            subsets = {'train': 0, 'test': 1}
            # filter unused videos and videos which do not belong to our train/test subset:
            used_videos = a2d_data_info[
                ~a2d_data_info.vid.isin(unused_videos)
                & (a2d_data_info.subset == subsets[subset])]
            used_videos_ids = list(used_videos['vid'])
            text_annotations = pandas.read_csv(
                osp.join(root_path, 'text_annotations/annotation.txt'))
            # filter the text annotations based on the used videos:
            used_text_annotations = text_annotations[
                text_annotations.video_id.isin(used_videos_ids)]
            # remove a single dataset annotation mistake in video: T6bNPuKV-wY
            used_text_annotations = used_text_annotations[
                used_text_annotations['instance_id'] != '1 (copy)']
            # convert data-frame to list of tuples:
            used_text_annotations = list(
                used_text_annotations.to_records(index=False))
            text_annotations_by_frame = []
            mask_annotations_dir = osp.join(
                root_path, 'text_annotations/annotation_with_instances')
            for video_id, instance_id, text_query in tqdm(
                    used_text_annotations):
                frame_annot_paths = sorted(
                    glob(osp.join(mask_annotations_dir, video_id, '*.h5')))
                instance_id = int(instance_id)
                for p in frame_annot_paths:
                    f = h5py.File(p)
                    instances = list(f['instance'])
                    if instance_id in instances:
                        # in case this instance does not appear in this frame it has no ground-truth mask, and thus this
                        # frame-instance pair is ignored in evaluation, same as SOTA method: CMPC-V. check out:
                        # https://github.com/spyflying/CMPC-Refseg/blob/094639b8bf00cc169ea7b49cdf9c87fdfc70d963/CMPC_video/build_A2D_batches.py#L98
                        frame_idx = int(p.split('/')[-1].split('.')[0])
                        text_query = text_query.lower(
                        )  # lower the text query prior to augmentation & tokenization
                        text_annotations_by_frame.append(
                            (text_query, video_id, frame_idx, instance_id))
            with open(saved_annotations_file_path, 'w') as f:
                json.dump(text_annotations_by_frame, f)
        if distributed:
            dist.barrier()
            with open(saved_annotations_file_path, 'r', encoding='utf-8') as f:
                text_annotations_by_frame = [tuple(a) for a in json.load(f)]
        return text_annotations_by_frame


class A2dSentencesTransforms:

    def __init__(self, subset_type, horizontal_flip_augmentations,
                 resize_and_crop_augmentations, train_short_size,
                 train_max_size, eval_short_size, eval_max_size, **kwargs):
        self.h_flip_augmentation = subset_type == 'train' and horizontal_flip_augmentations
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        scales = [
            train_short_size
        ]  # no more scales for now due to GPU memory constraints. might be changed later
        transforms = []
        if resize_and_crop_augmentations:
            if subset_type == 'train':
                transforms.append(
                    T.RandomResize(scales, max_size=train_max_size))
            elif subset_type == 'test':
                transforms.append(
                    T.RandomResize([eval_short_size], max_size=eval_max_size)),
        transforms.extend([T.ToTensor(), normalize])
        self.size_transforms = T.Compose(transforms)

    def __call__(self, source_frames, targets, text_query):
        if self.h_flip_augmentation and torch.rand(1) > 0.5:
            source_frames = [F.hflip(f) for f in source_frames]
            targets[len(targets) // 2]['masks'] = F.hflip(
                targets[len(targets) // 2]['masks'])
            # Note - is it possible for both 'right' and 'left' to appear together in the same query. hence this fix:
            text_query = text_query.replace('left', '@').replace(
                'right', 'left').replace('@', 'right')
        source_frames, targets = list(
            zip(*[
                self.size_transforms(f, t)
                for f, t in zip(source_frames, targets)
            ]))
        source_frames = torch.stack(source_frames)  # [T, 3, H, W]
        return source_frames, targets, text_query


class Collator:

    def __call__(self, batch):
        samples, targets, text_queries = list(zip(*batch))
        samples = nested_tensor_from_videos_list(samples)  # [T, B, C, H, W]
        # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
        targets = list(zip(*targets))
        batch_dict = {
            'samples': samples,
            'targets': targets,
            'text_queries': text_queries
        }
        return batch_dict


def get_text_annotations_gt(root_path, subset):
    # without 'header == None' pandas will ignore the first sample...
    a2d_data_info = pandas.read_csv(
        osp.join(root_path, 'Release/videoset.csv'), header=None)
    # 'vid', 'label', 'start_time', 'end_time', 'height', 'width', 'total_frames', 'annotated_frames', 'subset'
    a2d_data_info.columns = ['vid', '', '', '', '', '', '', '', 'subset']
    with open(
            osp.join(root_path, 'text_annotations/missed_videos.txt'),
            'r',
            encoding='utf-8') as f:
        unused_videos = f.read().splitlines()
    subsets = {'train': 0, 'test': 1}
    # filter unused videos and videos which do not belong to our train/test subset:
    used_videos = a2d_data_info[~a2d_data_info.vid.isin(unused_videos)
                                & (a2d_data_info.subset == subsets[subset])]
    used_videos_ids = list(used_videos['vid'])
    text_annotations = pandas.read_csv(
        osp.join(root_path, 'text_annotations/annotation.txt'))
    # filter the text annotations based on the used videos:
    used_text_annotations = text_annotations[text_annotations.video_id.isin(
        used_videos_ids)]
    # convert data-frame to list of tuples:
    used_text_annotations = list(used_text_annotations.to_records(index=False))
    return used_text_annotations


def create_a2d_sentences_ground_truth_test_annotations(dataset_path,
                                                       subset_type,
                                                       mask_annotations_dir,
                                                       output_path):
    text_annotations = get_text_annotations_gt(dataset_path, subset_type)

    # Note - it is very important to start counting the instance and category ids from 1 (not 0). This is implicitly
    # expected by pycocotools as it is the convention of the original coco dataset annotations.

    categories_dict = [{
        'id': 1,
        'name': 'dummy_class'
    }]  # dummy class, as categories are not used/predicted in RVOS

    images_dict = []
    annotations_dict = []
    images_set = set()
    instance_id_counter = 1
    for annot in tqdm(text_annotations):
        video_id, instance_id, text_query = annot
        annot_paths = sorted(
            glob(osp.join(mask_annotations_dir, video_id, '*.h5')))
        for p in annot_paths:
            f = h5py.File(p)
            instances = list(f['instance'])
            try:
                instance_idx = instances.index(int(instance_id))
            # in case this instance does not appear in this frame it has no ground-truth mask, and thus this
            # frame-instance pair is ignored in evaluation, same as SOTA method: CMPC-V. check out:
            # https://github.com/spyflying/CMPC-Refseg/blob/094639b8bf00cc169ea7b49cdf9c87fdfc70d963/CMPC_video/build_A2D_batches.py#L98
            except ValueError:
                continue  # instance_id does not appear in current frame
            mask = f['reMask'][instance_idx] if len(
                instances) > 1 else np.array(f['reMask'])
            mask = mask.transpose()

            frame_idx = int(p.split('/')[-1].split('.')[0])
            image_id = get_image_id(video_id, frame_idx, instance_id)
            assert image_id not in images_set, f'error: image id: {image_id} appeared twice'
            images_set.add(image_id)
            images_dict.append({
                'id': image_id,
                'height': mask.shape[0],
                'width': mask.shape[1]
            })

            mask_rle = encode(mask)
            mask_rle['counts'] = mask_rle['counts'].decode('ascii')
            mask_area = float(area(mask_rle))
            bbox = f['reBBox'][:, instance_idx] if len(
                instances) > 1 else np.array(
                    f['reBBox']).squeeze()  # x1y1x2y2 form
            bbox_xywh = [
                bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            ]
            instance_annot = {
                'id': instance_id_counter,
                'image_id': image_id,
                'category_id':
                1,  # dummy class, as categories are not used/predicted in ref-vos
                'segmentation': mask_rle,
                'area': mask_area,
                'bbox': bbox_xywh,
                'iscrowd': 0,
            }
            annotations_dict.append(instance_annot)
            instance_id_counter += 1
    dataset_dict = {
        'categories': categories_dict,
        'images': images_dict,
        'annotations': annotations_dict
    }
    with open(output_path, 'w') as f:
        json.dump(dataset_dict, f)
