# Part of the implementation is borrowed and modified from DUTCode,
# publicly available at https://github.com/Annbless/DUTCode

import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from modelscope.models.cv.video_stabilization.utils.MedianFilter import \
    MedianPool2d
from modelscope.models.cv.video_stabilization.utils.ProjectionUtils import (
    multiHomoEstimate, singleHomoEstimate)
from .config import cfg


class MotionPro(nn.Module):

    def __init__(self,
                 inplanes=2,
                 embeddingSize=64,
                 hiddenSize=128,
                 number_points=512,
                 kernel=5,
                 globalchoice='multi'):

        super(MotionPro, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(inplanes, embeddingSize, 1),
            nn.ReLU(),
        )
        self.embedding_motion = nn.Sequential(
            nn.Conv1d(inplanes, embeddingSize, 1),
            nn.ReLU(),
        )

        self.pad = kernel // 2
        self.conv1 = nn.Conv1d(embeddingSize, embeddingSize, 1)
        self.conv2 = nn.Conv1d(embeddingSize, embeddingSize // 2, 1)
        self.conv3 = nn.Conv1d(embeddingSize // 2, 1, 1)

        self.weighted = nn.Softmax(dim=2)

        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU(0.1)

        self.m_conv1 = nn.Conv1d(embeddingSize, 2 * embeddingSize, 1)
        self.m_conv2 = nn.Conv1d(2 * embeddingSize, 2 * embeddingSize, 1)
        self.m_conv3 = nn.Conv1d(2 * embeddingSize, embeddingSize, 1)

        self.fuse_conv1 = nn.Conv1d(embeddingSize + embeddingSize // 2,
                                    embeddingSize, 1)
        self.fuse_conv2 = nn.Conv1d(embeddingSize, embeddingSize, 1)

        self.decoder = nn.Linear(embeddingSize, 2, bias=False)

        if globalchoice == 'multi':
            self.homoEstimate = multiHomoEstimate
        elif globalchoice == 'single':
            self.homoEstimate = singleHomoEstimate

        self.meidanPool = MedianPool2d(5, same=True)

    def forward(self, motion):
        '''
        @param: motion contains distance info and motion info of keypoints

        @return: return predicted motion for each grid vertex
        '''
        distance_info = motion[:, 0:2, :]
        motion_info = motion[0:1, 2:4, :]

        embedding_distance = self.embedding(distance_info)
        embedding_distance = self.leakyRelu(self.conv1(embedding_distance))
        embedding_distance = self.leakyRelu(self.conv2(embedding_distance))
        distance_weighted = self.weighted(self.conv3(embedding_distance))

        embedding_motion = self.embedding_motion(motion_info)
        embedding_motion = self.leakyRelu(self.m_conv1(embedding_motion))
        embedding_motion = self.leakyRelu(self.m_conv2(embedding_motion))
        embedding_motion = self.leakyRelu(self.m_conv3(embedding_motion))
        embedding_motion = embedding_motion.repeat(distance_info.shape[0], 1,
                                                   1)

        embedding_motion = torch.cat([embedding_motion, embedding_distance], 1)
        embedding_motion = self.leakyRelu(self.fuse_conv1(embedding_motion))
        embedding_motion = self.leakyRelu(self.fuse_conv2(embedding_motion))

        embedding_motion = torch.sum(embedding_motion * distance_weighted, 2)

        out_motion = self.decoder(embedding_motion)

        return out_motion

    def inference(self, x_flow, y_flow, kp):
        """
        @param x_flow [B, 1, H, W]
        @param y_flow [B, 1, H, W]
        @param kp     [B*topk, 4 / 2]->[N, 4/2]
        """
        if kp.shape[1] == 4:
            kp = kp[:, 2:]
        index = kp.long()
        origin_motion = torch.cat([x_flow, y_flow], 1)
        extracted_motion = origin_motion[0, :, index[:, 0], index[:, 1]]
        kp = kp.permute(1, 0).float()
        concat_motion = torch.cat([kp[1:2, :], kp[0:1, :], extracted_motion],
                                  0)

        motion, gridsMotion, _ = self.homoEstimate(concat_motion, kp)
        GridMotion = (self.forward(motion)
                      + gridsMotion.squeeze(-1)) * cfg.MODEL.FLOWC
        GridMotion = GridMotion.view(cfg.MODEL.HEIGHT // cfg.MODEL.PIXELS,
                                     cfg.MODEL.WIDTH // cfg.MODEL.PIXELS, 2)
        GridMotion = GridMotion.permute(2, 0, 1).unsqueeze(0)
        GridMotion = self.meidanPool(GridMotion)
        return GridMotion


if __name__ == '__main__':
    model = MotionPro()
    model.train()
    model.cuda()
    x = torch.from_numpy(np.random.randn(4, 512).astype(np.float32)).cuda()
    kp = torch.from_numpy(np.random.randn(512, 2).astype(np.float32)).cuda()
    model.train_step(x, kp)
