# The implementation here is modified based on BaSSL,
# originally Apache 2.0 License and publicly avaialbe at https://github.com/kakaobrain/bassl

import math
import os
import os.path as osp
from typing import Any, Dict

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image
from shotdetect_scenedetect_lgss import shot_detect

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .get_model import get_contextual_relation_network, get_shot_encoder
from .utils.save_op import get_pred_boundary, pred2scene, scene2video

logger = get_logger()


@MODELS.register_module(
    Tasks.movie_scene_segmentation, module_name=Models.resnet50_bert)
class MovieSceneSegmentationModel(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, *args, **kwargs)

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        params = torch.load(model_path, map_location='cpu')

        config_path = osp.join(model_dir, ModelFile.CONFIGURATION)
        self.cfg = Config.from_file(config_path)

        def load_param_with_prefix(prefix, model, src_params):
            own_state = model.state_dict()
            for name, param in own_state.items():
                src_name = prefix + '.' + name
                own_state[name] = src_params[src_name]

            model.load_state_dict(own_state)

        self.shot_encoder = get_shot_encoder(self.cfg)
        load_param_with_prefix('shot_encoder', self.shot_encoder, params)
        self.crn = get_contextual_relation_network(self.cfg)
        load_param_with_prefix('crn', self.crn, params)

        crn_name = self.cfg.model.contextual_relation_network.name
        hdim = self.cfg.model.contextual_relation_network.params[crn_name][
            'hidden_size']
        self.head_sbd = nn.Linear(hdim, 2)
        load_param_with_prefix('head_sbd', self.head_sbd, params)

        self.test_transform = TF.Compose([
            TF.Resize(size=256, interpolation=Image.BICUBIC),
            TF.CenterCrop(224),
            TF.ToTensor(),
            TF.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        sampling_method = self.cfg.dataset.sampling_method.name
        self.neighbor_size = self.cfg.dataset.sampling_method.params[
            sampling_method].neighbor_size

        self.eps = 1e-5

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        data = inputs.pop('video')
        labels = inputs['label']
        outputs = self.shared_step(data)

        loss = F.cross_entropy(
            outputs.squeeze(), labels.squeeze(), reduction='none')
        lpos = labels == 1
        lneg = labels == 0

        pp, nn = 1, 1
        wp = (pp / float(pp + nn)) * lpos / (lpos.sum() + self.eps)
        wn = (nn / float(pp + nn)) * lneg / (lneg.sum() + self.eps)
        w = wp + wn
        loss = (w * loss).sum()

        probs = torch.argmax(outputs, dim=1)

        re = dict(pred=probs, loss=loss)
        return re

    def inference(self, batch):
        logger.info('Begin scene detect ......')
        bs = self.cfg.pipeline.batch_size_per_gpu
        sids = batch['sid']
        inputs = batch['shot_feat']

        shot_num = len(sids)
        cnt = math.ceil(shot_num / bs)

        infer_sid, infer_pred = [], []
        infer_result = {}
        for i in range(cnt):
            start = i * bs
            end = (i + 1) * bs if (i + 1) * bs < shot_num else shot_num
            input_ = inputs[start:end]
            sid_ = sids[start:end]
            input_ = torch.stack(input_)
            outputs = self.shared_step(input_)  # shape [b,2]
            prob = F.softmax(outputs, dim=1)
            infer_sid.extend(sid_.cpu().detach().numpy())
            infer_pred.extend(prob[:, 1].cpu().detach().numpy())
        infer_result.update({'pred': np.stack(infer_pred)})
        infer_result.update({'sid': infer_sid})

        assert len(infer_result['sid']) == len(sids)
        assert len(infer_result['pred']) == len(inputs)
        return infer_result

    def shared_step(self, inputs):
        with torch.no_grad():
            # infer shot encoder
            shot_repr = self.extract_shot_representation(inputs)
            assert len(shot_repr.shape) == 3

        # infer CRN
        _, pooled = self.crn(shot_repr, mask=None)
        # infer boundary score
        pred = self.head_sbd(pooled)
        return pred

    def save_shot_feat(self, _repr):
        feat = _repr.float().cpu().numpy()
        pth = self.cfg.dataset.img_path + '/features'
        os.makedirs(pth)

        for idx in range(_repr.shape[0]):
            name = f'shot_{str(idx).zfill(4)}.npy'
            name = osp.join(pth, name)
            np.save(name, feat[idx])

    def extract_shot_representation(self,
                                    inputs: torch.Tensor) -> torch.Tensor:
        """ inputs [b s k c h w] -> output [b d] """
        assert len(inputs.shape) == 6  # (B Shot Keyframe C H W)
        b, s, k, c, h, w = inputs.shape
        inputs = einops.rearrange(inputs, 'b s k c h w -> (b s) k c h w', s=s)
        keyframe_repr = [self.shot_encoder(inputs[:, _k]) for _k in range(k)]
        # [k (b s) d] -> [(b s) d]
        shot_repr = torch.stack(keyframe_repr).mean(dim=0)

        shot_repr = einops.rearrange(shot_repr, '(b s) d -> b s d', s=s)
        return shot_repr

    def postprocess(self, inputs: Dict[str, Any], **kwargs):
        logger.info('Generate scene .......')

        pred_dict = inputs['feat']
        thres = self.cfg.pipeline.save_threshold

        anno_dict = get_pred_boundary(pred_dict, thres)
        scene_dict_lst, scene_list, shot_num, shot_dict_lst = pred2scene(
            self.shot2keyf, anno_dict)
        if self.cfg.pipeline.save_split_scene:
            re_dir = scene2video(inputs['input_video_pth'], scene_list, thres)
            print(f'Split scene video saved to {re_dir}')
        return len(scene_list), scene_dict_lst, shot_num, shot_dict_lst

    def preprocess(self, inputs):
        logger.info('Begin shot detect......')
        shot_keyf_lst, anno, shot2keyf = shot_detect(
            inputs, **self.cfg.preprocessor.shot_detect)
        logger.info('Shot detect done!')

        single_shot_feat, sid = [], []
        for idx, one_shot in enumerate(shot_keyf_lst):
            one_shot = [
                self.test_transform(one_frame) for one_frame in one_shot
            ]
            one_shot = torch.stack(one_shot, dim=0)
            single_shot_feat.append(one_shot)
            sid.append(idx)
        single_shot_feat = torch.stack(single_shot_feat, dim=0)
        shot_feat = []
        for idx, one_shot in enumerate(anno):
            shot_idx = int(one_shot['shot_id']) + np.arange(
                -self.neighbor_size, self.neighbor_size + 1)
            shot_idx = np.clip(shot_idx, 0, one_shot['num_shot'])
            _one_shot = single_shot_feat[shot_idx]
            shot_feat.append(_one_shot)
        self.shot2keyf = shot2keyf
        self.anno = anno
        return shot_feat, sid
