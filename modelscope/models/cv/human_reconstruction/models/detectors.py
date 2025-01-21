# The implementation here is modified based on Pytorch, originally BSD License and publicly available at
# https://github.com/pytorch/pytorch
import numpy as np
import torch


class FasterRCNN(object):
    ''' detect body
    COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    '''

    def __init__(self, ckpt=None, device='cuda:0'):
        """
        https://pytorch.org/docs/stable/torchvision/models.html#faster-r-cnn
        """
        import torchvision
        if ckpt is None:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True)
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=False)
            state_dict = torch.load(ckpt, map_location='cpu')
            self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def run(self, input):
        """
        return: detected box, [x1, y1, x2, y2]
        """
        prediction = self.model(input.to(self.device))[0]
        inds = (prediction['labels'] == 1) * (prediction['scores'] > 0.5)
        if len(inds) < 1:
            return None
        else:
            bbox = prediction['boxes'][inds][0].cpu().numpy()
            return bbox

    @torch.no_grad()
    def run_multi(self, input):
        """
        return: detected box, [x1, y1, x2, y2]
        """
        prediction = self.model(input.to(self.device))[0]
        inds = (prediction['labels'] == 1) * (prediction['scores'] > 0.9)
        if len(inds) < 1:
            return None
        else:
            bbox = prediction['boxes'][inds].cpu().numpy()
            return bbox
