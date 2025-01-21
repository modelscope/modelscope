# Part of the implementation is borrowed and modified from MTTR,
# publicly available at https://github.com/mttr2021/MTTR

import os.path as osp
from typing import Any, Dict

import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.compatible_with_transformers import \
    compatible_position_ids
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .utils import (MTTR, A2DSentencesPostProcess, HungarianMatcher,
                    ReferYoutubeVOSPostProcess, SetCriterion,
                    flatten_temporal_batch_dims,
                    nested_tensor_from_videos_list)

logger = get_logger()


@MODELS.register_module(
    Tasks.referring_video_object_segmentation,
    module_name=Models.referring_video_object_segmentation)
class ReferringVideoObjectSegmentation(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, *args, **kwargs)

        config_path = osp.join(model_dir, ModelFile.CONFIGURATION)
        self.cfg = Config.from_file(config_path)
        transformer_cfg_dir = osp.join(model_dir, 'transformer_cfg_dir')

        self.model = MTTR(
            transformer_cfg_dir=transformer_cfg_dir, **self.cfg.model)

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        params_dict = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in params_dict.keys():
            params_dict = params_dict['model_state_dict']
        compatible_position_ids(
            params_dict, 'transformer.text_encoder.embeddings.position_ids')
        self.model.load_state_dict(params_dict, strict=True)

        self.set_postprocessor(self.cfg.pipeline.dataset_name)
        self.set_criterion()

    def set_device(self, device, name):
        self.device = device
        self._device_name = name

    def set_postprocessor(self, dataset_name):
        if 'a2d_sentences' in dataset_name or 'jhmdb_sentences' in dataset_name:
            self.postprocessor = A2DSentencesPostProcess()  # fine-tune
        elif 'ref_youtube_vos' in dataset_name:
            self.postprocessor = ReferYoutubeVOSPostProcess()  # inference
        else:
            assert False, f'postprocessing for dataset: {dataset_name} is not supported'

    def forward(self, inputs: Dict[str, Any]):
        samples = inputs['samples']
        targets = inputs['targets']
        text_queries = inputs['text_queries']

        valid_indices = torch.tensor(
            [i for i, t in enumerate(targets) if None not in t])
        targets = [targets[i] for i in valid_indices.tolist()]
        if self._device_name == 'gpu':
            samples = samples.to(self.device)
            valid_indices = valid_indices.to(self.device)
        if isinstance(text_queries, tuple):
            text_queries = list(text_queries)

        outputs = self.model(samples, valid_indices, text_queries)
        losses = -1
        if self.training:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k]
                         for k in loss_dict.keys() if k in weight_dict)

        predictions = []
        if not self.training:
            outputs.pop('aux_outputs', None)
            outputs, targets = flatten_temporal_batch_dims(outputs, targets)
            processed_outputs = self.postprocessor(
                outputs,
                resized_padded_sample_size=samples.tensors.shape[-2:],
                resized_sample_sizes=[t['size'] for t in targets],
                orig_sample_sizes=[t['orig_size'] for t in targets])
            image_ids = [t['image_id'] for t in targets]
            predictions = []
            for p, image_id in zip(processed_outputs, image_ids):
                for s, m in zip(p['scores'], p['rle_masks']):
                    predictions.append({
                        'image_id': image_id,
                        'category_id':
                        1,  # dummy label, as categories are not predicted in ref-vos
                        'segmentation': m,
                        'score': s.item()
                    })

        re = dict(pred=predictions, loss=losses)
        return re

    def inference(self, **kwargs):
        window = kwargs['window']
        text_query = kwargs['text_query']
        video_metadata = kwargs['metadata']

        window = nested_tensor_from_videos_list([window])
        valid_indices = torch.arange(len(window.tensors))
        if self._device_name == 'gpu':
            valid_indices = valid_indices.cuda()
        outputs = self.model(window, valid_indices, [text_query])
        window_masks = self.postprocessor(
            outputs, [video_metadata],
            window.tensors.shape[-2:])[0]['pred_masks']
        return window_masks

    def postprocess(self, inputs: Dict[str, Any], **kwargs):
        return inputs

    def set_criterion(self):
        matcher = HungarianMatcher(
            cost_is_referred=self.cfg.matcher.set_cost_is_referred,
            cost_dice=self.cfg.matcher.set_cost_dice)
        weight_dict = {
            'loss_is_referred': self.cfg.loss.is_referred_loss_coef,
            'loss_dice': self.cfg.loss.dice_loss_coef,
            'loss_sigmoid_focal': self.cfg.loss.sigmoid_focal_loss_coef
        }

        if self.cfg.loss.aux_loss:
            aux_weight_dict = {}
            for i in range(self.cfg.model.num_decoder_layers - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v
                     for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        self.criterion = SetCriterion(
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=self.cfg.loss.eos_coef)
