# Part of the implementation is borrowed and modified from CRNN,
# publicly available at https://github.com/meijieru/crnn.pytorch
# paper linking at https://arxiv.org/pdf/1507.05717.pdf
import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):

    def __init__(self):
        super(CRNN, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.p0 = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.p1 = nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.p2 = nn.MaxPool2d(
            kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                256, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.p3 = nn.MaxPool2d(
            kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                512, 512, kernel_size=(2, 1), padding=(0, 0), stride=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256), BidirectionalLSTM(256, 256, 512))

        self.cls = nn.Linear(512, 7644, bias=False)

    def forward(self, input):
        # RGB2GRAY
        input = input[:, 0:
                      1, :, :] * 0.2989 + input[:, 1:
                                                2, :, :] * 0.5870 + input[:, 2:
                                                                          3, :, :] * 0.1140
        feats = self.conv0(input)
        feats = self.p0(feats)
        feats = self.conv1(feats)
        feats = self.p1(feats)
        feats = self.conv2(feats)
        feats = self.p2(feats)
        feats = self.conv3(feats)
        feats = self.p3(feats)
        convfeats = self.conv4(feats)

        b, c, h, w = convfeats.size()
        assert h == 1, 'the height of conv must be 1'
        convfeats = convfeats.squeeze(2)
        convfeats = convfeats.permute(2, 0, 1)  # [w, b, c]

        rnnfeats = self.rnn(convfeats)
        output = self.cls(rnnfeats)
        output = output.permute(1, 0, 2)  # [b, w, c]
        return output
