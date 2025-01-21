# Copyright (c) Alibaba, Inc. and its affiliates.

import ast
import io
from typing import Any, Dict, List, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

from modelscope.fileio import File
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import InputModel, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['SegmentationClusteringPipeline']


@PIPELINES.register_module(
    Tasks.speaker_diarization, module_name=Pipelines.segmentation_clustering)
class SegmentationClusteringPipeline(Pipeline):
    """Segmentation and Clustering Pipeline
    use `model` to create a Segmentation and Clustering Pipeline.

    Args:
        model (SegmentationClusteringPipeline): A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the pipeline's constructor.
    Example:
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> p = pipeline(
    >>>    task=Tasks.speaker_diarization, model='damo/speech_campplus_speaker-diarization_common')
    >>> print(p(audio))

    """

    def __init__(self, model: InputModel, **kwargs):
        """use `model` to create a speaker diarization pipeline for prediction
        Args:
            model (str): a valid offical model id
        """
        super().__init__(model=model, **kwargs)
        self.config = self.model.other_config
        config = {
            'seg_dur': 1.5,
            'seg_shift': 0.75,
        }
        self.config.update(config)
        self.fs = self.config['sample_rate']
        self.sv_pipeline = pipeline(
            task='speaker-verification', model=self.config['speaker_model'])

    def __call__(self, audio: Union[str, np.ndarray, list],
                 **params) -> Dict[str, Any]:
        """ extract the speaker embeddings of input audio and do cluster
        Args:
            audio (str, np.ndarray, list): If it is represented as a str or a np.ndarray, it
            should be a complete speech signal and requires VAD preprocessing. If the audio
            is represented as a list, it should contain only the effective speech segments
            obtained through VAD preprocessing. The list should be formatted as [[0(s),3.2,
            np.ndarray], [5.3,9.1, np.ndarray], ...]. Each element is a sublist that contains
            the start time, end time, and the numpy array of the speech segment respectively.
        """
        self.config.update(params)
        # vad
        logger.info('Doing VAD...')
        vad_segments = self.preprocess(audio)
        # check input data
        self.check_audio_list(vad_segments)
        # segmentation
        logger.info('Doing segmentation...')
        segments = self.chunk(vad_segments)
        # embedding
        logger.info('Extracting embeddings...')
        embeddings = self.forward(segments)
        # clustering
        logger.info('Clustering...')
        labels = self.clustering(embeddings)
        # post processing
        logger.info('Post processing...')
        output = self.postprocess(segments, vad_segments, labels, embeddings)
        return {OutputKeys.TEXT: output}

    def forward(self, input: list) -> np.ndarray:
        embeddings = []
        for s in input:
            save_dict = self.sv_pipeline([s[2]], output_emb=True)
            if save_dict['embs'].shape == (1, 192):
                embeddings.append(save_dict['embs'])
        embeddings = np.concatenate(embeddings)
        return embeddings

    def clustering(self, embeddings: np.ndarray) -> np.ndarray:
        labels = self.model(embeddings, **self.config)
        return labels

    def postprocess(self, segments: list, vad_segments: list,
                    labels: np.ndarray, embeddings: np.ndarray) -> list:
        assert len(segments) == len(labels)
        labels = self.correct_labels(labels)
        distribute_res = []
        for i in range(len(segments)):
            distribute_res.append([segments[i][0], segments[i][1], labels[i]])
        # merge the same speakers chronologically
        distribute_res = self.merge_seque(distribute_res)

        # accquire speaker center
        spk_embs = []
        for i in range(labels.max() + 1):
            spk_emb = embeddings[labels == i].mean(0)
            spk_embs.append(spk_emb)
        spk_embs = np.stack(spk_embs)

        def is_overlapped(t1, t2):
            if t1 > t2 + 1e-4:
                return True
            return False

        # distribute the overlap region
        for i in range(1, len(distribute_res)):
            if is_overlapped(distribute_res[i - 1][1], distribute_res[i][0]):
                p = (distribute_res[i][0] + distribute_res[i - 1][1]) / 2
                if 'change_locator' in self.config:
                    if not hasattr(self, 'change_locator_pipeline'):
                        self.change_locator_pipeline = pipeline(
                            task=Tasks.speaker_diarization,
                            model=self.config['change_locator'])
                    short_utt_st = max(p - 1.5, distribute_res[i - 1][0])
                    short_utt_ed = min(p + 1.5, distribute_res[i][1])
                    if short_utt_ed - short_utt_st > 1:
                        audio_data = self.cut_audio(short_utt_st, short_utt_ed,
                                                    vad_segments)
                        spk1 = distribute_res[i - 1][2]
                        spk2 = distribute_res[i][2]
                        _, ct = self.change_locator_pipeline(
                            audio_data, [spk_embs[spk1], spk_embs[spk2]],
                            output_res=True)
                        if ct is not None:
                            p = short_utt_st + ct
                distribute_res[i][0] = p
                distribute_res[i - 1][1] = p

        # smooth the result
        distribute_res = self.smooth(distribute_res)

        return distribute_res

    def preprocess(self, audio: Union[str, np.ndarray, list]) -> list:
        if isinstance(audio, list):
            audio.sort(key=lambda x: x[0])
            return audio
        elif isinstance(audio, str):
            file_bytes = File.read(audio)
            audio, fs = sf.read(io.BytesIO(file_bytes), dtype='float32')
            if len(audio.shape) == 2:
                audio = audio[:, 0]
            if fs != self.fs:
                logger.info(
                    f'[WARNING]: The sample rate of audio is not {self.fs}, resample it.'
                )
                audio, fs = torchaudio.sox_effects.apply_effects_tensor(
                    torch.from_numpy(audio).unsqueeze(0),
                    fs,
                    effects=[['rate', str(self.fs)]])
                audio = audio.squeeze(0).numpy()
        assert len(audio.shape) == 1, 'modelscope error: Wrong audio format.'
        if audio.dtype in ['int16', 'int32', 'int64']:
            audio = (audio / (1 << 15)).astype('float32')
        else:
            audio = audio.astype('float32')
        if not hasattr(self, 'vad_pipeline'):
            self.vad_pipeline = pipeline(
                task=Tasks.voice_activity_detection,
                model=self.config['vad_model'],
                model_revision='v2.0.2')
        vad_time = self.vad_pipeline(
            audio, fs=self.fs, is_final=True)[0]['value']
        vad_segments = []
        if isinstance(vad_time, str):
            vad_time_list = ast.literal_eval(vad_time)
        elif isinstance(vad_time, list):
            vad_time_list = vad_time
        else:
            raise ValueError('Incorrect vad result. Get %s' % (type(vad_time)))
        for t in vad_time_list:
            st = int(t[0]) / 1000
            ed = int(t[1]) / 1000
            vad_segments.append(
                [st, ed, audio[int(st * self.fs):int(ed * self.fs)]])

        return vad_segments

    def check_audio_list(self, audio: list):
        audio_dur = 0
        for i in range(len(audio)):
            seg = audio[i]
            assert seg[1] >= seg[0], 'modelscope error: Wrong time stamps.'
            assert isinstance(seg[2],
                              np.ndarray), 'modelscope error: Wrong data type.'
            assert int(seg[1] * self.fs) - int(
                seg[0] * self.fs
            ) == seg[2].shape[
                0], 'modelscope error: audio data in list is inconsistent with time length.'
            if i > 0:
                assert seg[0] >= audio[
                    i - 1][1], 'modelscope error: Wrong time stamps.'
            audio_dur += seg[1] - seg[0]
        assert audio_dur > 5, 'modelscope error: The effective audio duration is too short.'

    def chunk(self, vad_segments: list) -> list:

        def seg_chunk(seg_data):
            seg_st = seg_data[0]
            data = seg_data[2]
            chunk_len = int(self.config['seg_dur'] * self.fs)
            chunk_shift = int(self.config['seg_shift'] * self.fs)
            last_chunk_ed = 0
            seg_res = []
            for chunk_st in range(0, data.shape[0], chunk_shift):
                chunk_ed = min(chunk_st + chunk_len, data.shape[0])
                if chunk_ed <= last_chunk_ed:
                    break
                last_chunk_ed = chunk_ed
                chunk_st = max(0, chunk_ed - chunk_len)
                chunk_data = data[chunk_st:chunk_ed]
                if chunk_data.shape[0] < chunk_len:
                    chunk_data = np.pad(chunk_data,
                                        (0, chunk_len - chunk_data.shape[0]),
                                        'constant')
                seg_res.append([
                    chunk_st / self.fs + seg_st, chunk_ed / self.fs + seg_st,
                    chunk_data
                ])
            return seg_res

        segs = []
        for i, s in enumerate(vad_segments):
            segs.extend(seg_chunk(s))

        return segs

    def cut_audio(self, cut_st: float, cut_ed: float,
                  audio: Union[np.ndarray, list]) -> np.ndarray:
        # collect audio data given the start and end time.
        if isinstance(audio, np.ndarray):
            return audio[int(cut_st * self.fs):int(cut_ed * self.fs)]
        elif isinstance(audio, list):
            for i in range(len(audio)):
                if i == 0:
                    if cut_st < audio[i][1]:
                        st_i = i
                else:
                    if cut_st >= audio[i - 1][1] and cut_st < audio[i][1]:
                        st_i = i

                if i == len(audio) - 1:
                    if cut_ed > audio[i][0]:
                        ed_i = i
                else:
                    if cut_ed > audio[i][0] and cut_ed <= audio[i + 1][0]:
                        ed_i = i
            audio_segs = audio[st_i:ed_i + 1]
            cut_data = []
            for i in range(len(audio_segs)):
                s_st, s_ed, data = audio_segs[i]
                cut_data.append(
                    data[int((max(cut_st, s_st) - s_st)
                             * self.fs):int((min(cut_ed, s_ed) - s_st)
                                            * self.fs)])
            cut_data = np.concatenate(cut_data)
            return cut_data
        else:
            raise ValueError('modelscope error: Wrong audio format.')

    def correct_labels(self, labels):
        labels_id = 0
        id2id = {}
        new_labels = []
        for i in labels:
            if i not in id2id:
                id2id[i] = labels_id
                labels_id += 1
            new_labels.append(id2id[i])
        return np.array(new_labels)

    def merge_seque(self, distribute_res):
        res = [distribute_res[0]]
        for i in range(1, len(distribute_res)):
            if distribute_res[i][2] != res[-1][2] or distribute_res[i][
                    0] > res[-1][1]:
                res.append(distribute_res[i])
            else:
                res[-1][1] = distribute_res[i][1]
        return res

    def smooth(self, res, mindur=1):
        # short segments are assigned to nearest speakers.
        for i in range(len(res)):
            res[i][0] = round(res[i][0], 2)
            res[i][1] = round(res[i][1], 2)
            if res[i][1] - res[i][0] < mindur:
                if i == 0:
                    res[i][2] = res[i + 1][2]
                elif i == len(res) - 1:
                    res[i][2] = res[i - 1][2]
                elif res[i][0] - res[i - 1][1] <= res[i + 1][0] - res[i][1]:
                    res[i][2] = res[i - 1][2]
                else:
                    res[i][2] = res[i + 1][2]
        # merge the speakers
        res = self.merge_seque(res)

        return res
