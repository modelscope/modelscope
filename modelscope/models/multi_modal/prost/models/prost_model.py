# The implementation is adopted from the CLIP4Clip implementation,
# made publicly available under Apache License, Version 2.0 at https://github.com/ArrowLuo/CLIP4Clip

import os
import random
import uuid
from os.path import exists
from tempfile import TemporaryDirectory
from typing import Any, Dict
from urllib.parse import urlparse

import json
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

from modelscope.hub.file_download import http_get_file
from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.prost.models.modeling import CLIP4Clip
from modelscope.models.multi_modal.prost.models.tokenization_clip import \
    SimpleTokenizer as ClipTokenizer
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from ..dataloaders.rawvideo_util import RawVideoExtractor

logger = get_logger()


@MODELS.register_module(Tasks.text_video_retrieval, module_name=Models.prost)
class ProSTForTVRetrieval(TorchModel):

    def __init__(self, model_dir, **kwargs):
        super().__init__(model_dir=model_dir, **kwargs)
        # model config parameters
        with open(
                f'{model_dir}/{ModelFile.CONFIGURATION}', 'r',
                encoding='utf-8') as json_file:
            all_model_config = json.load(json_file)
        model_config = all_model_config['paras']

        cross_model_config = all_model_config['crossbase']
        # print(all_model_config)
        # print(cross_model_config)
        model_config['model_dir'] = model_dir
        self.SPECIAL_TOKEN = {
            'CLS_TOKEN': '<|startoftext|>',
            'SEP_TOKEN': '<|endoftext|>',
            'MASK_TOKEN': '[MASK]',
            'UNK_TOKEN': '[UNK]',
            'PAD_TOKEN': '[PAD]'
        }
        self.max_words = model_config['max_words']
        self.max_frames = model_config['max_frames']
        self.feature_framerate = model_config['feature_framerate']
        self.image_resolution = 224
        if torch.cuda.is_available():
            self.device = model_config['device']
        else:
            self.device = 'cpu'
        self.init_model = f'{model_dir}/{ModelFile.TORCH_MODEL_BIN_FILE}'

        self.tokenizer = ClipTokenizer(model_dir)
        self.rawVideoExtractor = RawVideoExtractor(
            frame_rate=self.feature_framerate, size=self.image_resolution)
        self.local_transform = self.rawVideoExtractor.transform
        self.model = CLIP4Clip.from_pretrained(
            cross_config=cross_model_config, task_config=model_config)
        if hasattr(self.model, 'module'):
            self.model = self.model.module.to(self.device)
        else:
            self.model = self.model.to(self.device)
        if self.init_model:
            assert exists(self.init_model)
            model_state_dict = torch.load(self.init_model, map_location='cpu')
            self.model.load_state_dict(model_state_dict, strict=False)
        self.model.to(self.device)

    def _get_text(self, caption, tokenizer, enable_zh=False):

        if type(caption) is str:
            _caption_text, s, e = caption, None, None
        elif type(caption) is tuple:
            if len(caption) == 3:
                _caption_text, s, e = caption
            elif len(caption) == 4:
                _caption_text, s, e, pos = caption
            else:
                NotImplementedError

        if isinstance(_caption_text, list):
            caption_text = random.choice(_caption_text)
        else:
            caption_text = _caption_text
        if enable_zh:
            _token = tokenizer.encode(caption_text)
            input_ids = _token.ids
            input_mask = _token.attention_mask
            segment_ids = _token.type_ids
        else:
            words = tokenizer.tokenize(caption_text)

            words = [self.SPECIAL_TOKEN['CLS_TOKEN']] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN['SEP_TOKEN']]

            input_ids = tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words

        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, s, e

    def _get_rawvideo_dec(self,
                          video_path,
                          rawVideoExtractor,
                          local_transform,
                          s=None,
                          e=None):
        video_mask = np.zeros(self.max_frames, dtype=int)
        max_video_length = 0

        # T x 3 x H x W
        video = np.zeros((self.max_frames, 3, rawVideoExtractor.size,
                          rawVideoExtractor.size),
                         dtype=float)

        if s is None:
            start_time, end_time = None, None
        else:
            start_time = int(s)
            end_time = int(e)
            start_time = start_time if start_time >= 0. else 0.
            end_time = end_time if end_time >= 0. else 0.
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            elif start_time == end_time:
                end_time = end_time + 1

        url_parsed = urlparse(video_path)
        if url_parsed.scheme in ('file', '') and exists(
                url_parsed.path):  # Possibly a local file
            vreader = VideoReader(video_path, ctx=cpu(0))
        else:
            try:
                with TemporaryDirectory() as temporary_cache_dir:
                    random_str = uuid.uuid4().hex
                    http_get_file(
                        url=video_path,
                        local_dir=temporary_cache_dir,
                        file_name=random_str,
                        cookies=None)
                    temp_file_path = os.path.join(temporary_cache_dir,
                                                  random_str)
                    vreader = VideoReader(temp_file_path, ctx=cpu(0))
            except Exception as ex:
                logger.error('non video input, output is {}!!!'.format(ex))
                return video, video_mask

        fps = vreader.get_avg_fps()
        f_start = 0 if start_time is None else int(start_time * fps)
        f_end = int(
            min(1000000000 if end_time is None else end_time * fps,
                len(vreader) - 1))
        num_frames = f_end - f_start + 1
        if num_frames > 0:
            # L x T x 3 x H x W
            sample_fps = int(self.feature_framerate)
            t_stride = int(round(float(fps) / sample_fps))

            all_pos = list(range(f_start, f_end + 1, t_stride))
            if len(all_pos) > self.max_frames:
                sample_pos = [
                    all_pos[_] for _ in np.linspace(
                        0, len(all_pos) - 1, num=self.max_frames, dtype=int)
                ]
            else:
                sample_pos = all_pos
            patch_images = [
                Image.fromarray(f)
                for f in vreader.get_batch(sample_pos).asnumpy()
            ]
            patch_images = torch.stack(
                [local_transform(img) for img in patch_images])
            slice_len = patch_images.shape[0]
            max_video_length = max_video_length if max_video_length > slice_len else slice_len
            if slice_len < 1:
                pass
            else:
                video[:slice_len, ...] = patch_images
        else:
            logger.error('video path: {} error. video id: {}'.format(
                video_path, video_id))

        video_mask[:max_video_length] = [1] * max_video_length

        return video, video_mask

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        from modelscope.outputs import OutputKeys
        output = {}

        if 'video' in input and input['video'] is not None:
            video_path = input['video']
            video, video_mask = self._get_rawvideo_dec(video_path,
                                                       self.rawVideoExtractor,
                                                       self.local_transform)
            video = torch.unsqueeze(
                torch.from_numpy(video), dim=0).to(self.device)
            video_mask = torch.unsqueeze(
                torch.from_numpy(video_mask), dim=0).to(self.device)

        if 'text' in input and input['text'] is not None:
            caption = input['text']
            pairs_text, pairs_mask, pairs_segment, s, e = self._get_text(
                caption, self.tokenizer, enable_zh=False)
            input_ids = torch.unsqueeze(
                torch.from_numpy(pairs_text), dim=0).to(self.device)
            input_mask = torch.unsqueeze(
                torch.from_numpy(pairs_mask), dim=0).to(self.device)
            segment_ids = torch.unsqueeze(
                torch.from_numpy(pairs_segment), dim=0).to(self.device)

        phr_feat, sen_feat, obj_feat, eve_feat = self.model.get_sequence_visual_output(
            input_ids, segment_ids, input_mask, video, video_mask)

        sim_espm, _, sim_oppm, _ = self.model.get_max_similarity_logits(
            phr_feat,
            sen_feat,
            obj_feat,
            eve_feat,
            input_mask,
            video_mask,
            shaped=True)
        # logger.info('sim: {}'.format(sim_espm))
        # logger.info('sim: {}'.format(sim_oppm))
        sim_tv = sim_espm + 1.5 * sim_oppm

        # logger.info('phrase prototype: {}'.format(phr_feat.shape))
        # logger.info('sentence prototype: {}'.format(sen_feat.shape))
        # logger.info('object prototype: {}'.format(obj_feat.shape))
        # logger.info('event prototype: {}'.format(eve_feat.shape))
        output[OutputKeys.TEXTVIDEO_SIM] = sim_tv.cpu().detach().numpy()
        output[OutputKeys.PHRASE_PROTOTYPE] = phr_feat.cpu().detach().numpy()
        output[OutputKeys.SENTENCE_PROTOTYPE] = sen_feat.cpu().detach().numpy()
        output[OutputKeys.OBJECT_PROTOTYPE] = obj_feat.cpu().detach().numpy()
        output[OutputKeys.EVENT_PROTOTYPE] = eve_feat.cpu().detach().numpy()
        return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
