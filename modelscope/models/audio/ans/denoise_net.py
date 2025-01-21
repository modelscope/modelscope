# Copyright (c) Alibaba, Inc. and its affiliates.
# Related papers:
# Shengkui Zhao, Trung Hieu Nguyen, Bin Ma, “Monaural Speech Enhancement with Complex Convolutional
# Block Attention Module and Joint Time Frequency Losses”, ICASSP 2021.
# Shiliang Zhang, Ming Lei, Zhijie Yan, Lirong Dai, “Deep-FSMN for Large Vocabulary Continuous Speech
# Recognition “, arXiv:1803.05030, 2018.

from torch import nn

from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.models.audio.ans.layers.activations import (RectifiedLinear,
                                                            Sigmoid)
from modelscope.models.audio.ans.layers.affine_transform import AffineTransform
from modelscope.models.audio.ans.layers.uni_deep_fsmn import UniDeepFsmn
from modelscope.utils.constant import Tasks


@MODELS.register_module(
    Tasks.acoustic_noise_suppression, module_name=Models.speech_dfsmn_ans)
class DfsmnAns(TorchModel):
    """Denoise model with DFSMN.

    Args:
        model_dir (str): the model path.
        fsmn_depth (int): the depth of deepfsmn
        lorder (int):
    """

    def __init__(self,
                 model_dir: str,
                 fsmn_depth=9,
                 lorder=20,
                 *args,
                 **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.lorder = lorder
        self.linear1 = AffineTransform(120, 256)
        self.relu = RectifiedLinear(256, 256)
        repeats = [
            UniDeepFsmn(256, 256, lorder, 256) for i in range(fsmn_depth)
        ]
        self.deepfsmn = nn.Sequential(*repeats)
        self.linear2 = AffineTransform(256, 961)
        self.sig = Sigmoid(961, 961)

    def forward(self, input):
        """
        Args:
            input: fbank feature [batch_size,number_of_frame,feature_dimension]

        Returns:
            mask value [batch_size, number_of_frame, FFT_size/2+1]
        """
        x1 = self.linear1(input)
        x2 = self.relu(x1)
        x3 = self.deepfsmn(x2)
        x4 = self.linear2(x3)
        x5 = self.sig(x4)
        return x5

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<Nnet>\n'
        re_str += self.linear1.to_kaldi_nnet()
        re_str += self.relu.to_kaldi_nnet()
        for dfsmn in self.deepfsmn:
            re_str += dfsmn.to_kaldi_nnet()
        re_str += self.linear2.to_kaldi_nnet()
        re_str += self.sig.to_kaldi_nnet()
        re_str += '</Nnet>\n'

        return re_str
