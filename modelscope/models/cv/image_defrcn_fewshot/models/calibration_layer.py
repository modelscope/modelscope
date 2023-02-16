# The implementation is adopted from er-muyue/DeFRCN
# made publicly available under the MIT License at
# https://github.com/er-muyue/DeFRCN/blob/main/defrcn/evaluation/calibration_layer.py

import cv2
import torch
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import ImageList
from sklearn.metrics.pairwise import cosine_similarity

from modelscope.utils.logger import get_logger
from .resnet import resnet101


class DatasetMapperIns(DatasetMapper):

    def __init__(self, cfg, is_train: bool):
        super(DatasetMapperIns, self).__init__(cfg, is_train)

    def __call__(self, dataset_dict):
        is_train = self.is_train
        self.is_train = True
        dataset_dict = super(DatasetMapperIns, self).__call__(dataset_dict)
        self.is_train = is_train
        return dataset_dict


class PrototypicalCalibrationBlock:

    def __init__(self, cfg):
        super().__init__()

        self.logger = get_logger()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.alpha = self.cfg.TEST.PCB_ALPHA

        self.imagenet_model = self.build_model()
        self.dataloader = build_detection_test_loader(
            self.cfg,
            self.cfg.DATASETS.TRAIN[0],
            mapper=DatasetMapperIns(cfg, False))
        self.roi_pooler = ROIPooler(
            output_size=(1, 1),
            scales=(1 / 32, ),
            sampling_ratio=(0),
            pooler_type='ROIAlignV2')
        self.prototypes = self.build_prototypes()

        self.exclude_cls = self.clsid_filter()

    def build_model(self):
        self.logger.info('Loading ImageNet Pre-train Model from {}'.format(
            self.cfg.TEST.PCB_MODELPATH))
        if self.cfg.TEST.PCB_MODELTYPE == 'resnet':
            imagenet_model = resnet101()
        else:
            raise NotImplementedError
        state_dict = torch.load(self.cfg.TEST.PCB_MODELPATH)
        imagenet_model.load_state_dict(state_dict)
        imagenet_model = imagenet_model.to(self.device)
        imagenet_model.eval()
        return imagenet_model

    def build_prototypes(self):

        all_features, all_labels = [], []
        for index in range(len(self.dataloader.dataset)):
            inputs = [self.dataloader.dataset[index]]
            assert len(inputs) == 1
            # load support images and gt-boxes
            img = cv2.imread(inputs[0]['file_name'])  # BGR
            img_h, _ = img.shape[0], img.shape[1]
            ratio = img_h / inputs[0]['instances'].image_size[0]
            inputs[0]['instances'].gt_boxes.tensor = inputs[0][
                'instances'].gt_boxes.tensor * ratio
            boxes = [x['instances'].gt_boxes.to(self.device) for x in inputs]

            # extract roi features
            features = self.extract_roi_features(img, boxes)
            all_features.append(features.cpu().data)

            gt_classes = [x['instances'].gt_classes for x in inputs]
            all_labels.append(gt_classes[0].cpu().data)

        # concat
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        assert all_features.shape[0] == all_labels.shape[0]

        # calculate prototype
        features_dict = {}
        for i, label in enumerate(all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(all_features[i].unsqueeze(0))

        prototypes_dict = {}
        for label in features_dict:
            features = torch.cat(features_dict[label], dim=0)
            prototypes_dict[label] = torch.mean(features, dim=0, keepdim=True)

        return prototypes_dict

    def extract_roi_features(self, img, boxes):
        """
        :param img:
        :param boxes:
        :return:
        """

        mean = torch.tensor([0.406, 0.456, 0.485]).reshape(
            (3, 1, 1)).to(self.device)
        std = torch.tensor([[0.225, 0.224, 0.229]]).reshape(
            (3, 1, 1)).to(self.device)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        images = [(img / 255. - mean) / std]
        images = ImageList.from_tensors(images, 0)
        conv_feature = self.imagenet_model(
            images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW

        box_features = self.roi_pooler([conv_feature],
                                       boxes).squeeze(2).squeeze(2)

        activation_vectors = self.imagenet_model.fc(box_features)

        return activation_vectors

    def execute_calibration(self, inputs, dts):
        if 'file_name' in inputs[0]:
            img = cv2.imread(inputs[0]['file_name'])
        elif 'image_numpy' in inputs[0]:
            img = inputs[0]['image_numpy']

        ileft = (dts[0]['instances'].scores > self.cfg.TEST.PCB_UPPER).sum()
        iright = (dts[0]['instances'].scores > self.cfg.TEST.PCB_LOWER).sum()
        assert ileft <= iright
        boxes = [dts[0]['instances'].pred_boxes[ileft:iright]]

        features = self.extract_roi_features(img, boxes)

        for i in range(ileft, iright):
            tmp_class = int(dts[0]['instances'].pred_classes[i])
            if tmp_class in self.exclude_cls:
                continue
            tmp_cos = cosine_similarity(
                features[i - ileft].cpu().data.numpy().reshape((1, -1)),
                self.prototypes[tmp_class].cpu().data.numpy())[0][0]
            dts[0]['instances'].scores[i] = dts[0]['instances'].scores[
                i] * self.alpha + tmp_cos * (1 - self.alpha)
        return dts

    def clsid_filter(self):
        dsname = self.cfg.DATASETS.TEST[0]
        exclude_ids = []
        if 'test_all' in dsname:
            if 'coco' in dsname:
                exclude_ids = [
                    7, 9, 10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44,
                    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 59, 61, 63, 64,
                    65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79
                ]
            elif 'voc' in dsname:
                exclude_ids = list(range(0, 15))
            else:
                raise NotImplementedError
        return exclude_ids


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
