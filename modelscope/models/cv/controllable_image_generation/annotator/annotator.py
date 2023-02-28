# Part of the implementation is borrowed and modified from ControlNet,
# publicly available at https://github.com/lllyasviel/ControlNet
import os

import cv2
import mmcv
import numpy as np
import torch
from einops import rearrange
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette

from .midas.api import MiDaSInference
from .mlsd.mbv2_mlsd_large import MobileV2_MLSD_Large
from .mlsd.utils import pred_lines
from .openpose import util
from .openpose.body import Body
from .openpose.hand import Hand

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class OpenposeDetector:

    def __init__(self, annotator_ckpts_path, device='cuda'):
        body_modelpath = os.path.join(annotator_ckpts_path,
                                      'body_pose_model.pth')
        hand_modelpath = os.path.join(annotator_ckpts_path,
                                      'hand_pose_model.pth')

        self.body_estimation = Body(body_modelpath, device)
        self.hand_estimation = Hand(hand_modelpath, device)

    def __call__(self, oriImg, hand=False):
        oriImg = oriImg[:, :, ::-1].copy()
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            canvas = np.zeros_like(oriImg)
            canvas = util.draw_bodypose(canvas, candidate, subset)
            if hand:
                hands_list = util.handDetect(candidate, subset, oriImg)
                all_hand_peaks = []
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(oriImg[y:y + w, x:x + w, :])
                    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0],
                                           peaks[:, 0] + x)
                    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1],
                                           peaks[:, 1] + y)
                    all_hand_peaks.append(peaks)
                canvas = util.draw_handpose(canvas, all_hand_peaks)
            return canvas, dict(
                candidate=candidate.tolist(), subset=subset.tolist())


class MLSDdetector:

    def __init__(self, annotator_ckpts_path, device='cuda'):
        model_path = os.path.join(annotator_ckpts_path,
                                  'mlsd_large_512_fp32.pth')
        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path), strict=True)
        self.model = model.to(device).eval()

    def __call__(self, input_image, thr_v, thr_d):
        assert input_image.ndim == 3
        img = input_image
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = pred_lines(img, self.model,
                                   [img.shape[0], img.shape[1]], thr_v, thr_d)
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(img_output, (x_start, y_start), (x_end, y_end),
                             [255, 255, 255], 1)
        except Exception:
            pass
        return img_output[:, :, 0]


class MidasDetector:

    def __init__(self, model_root_path, device='cuda'):
        self.model = MiDaSInference(
            model_type='dpt_hybrid',
            model_root_path=model_root_path).to(device)

    def __call__(self, input_image, a=np.pi * 2.0, bg_th=0.1):
        assert input_image.ndim == 3
        image_depth = input_image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().cuda()
            image_depth = image_depth / 127.5 - 1.0
            image_depth = rearrange(image_depth, 'h w c -> 1 c h w')
            depth = self.model(image_depth)[0]

            depth_pt = depth.clone()
            depth_pt -= torch.min(depth_pt)
            depth_pt /= torch.max(depth_pt)
            depth_pt = depth_pt.cpu().numpy()
            depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)

            depth_np = depth.cpu().numpy()
            x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
            y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
            z = np.ones_like(x) * a
            x[depth_pt < bg_th] = 0
            y[depth_pt < bg_th] = 0
            normal = np.stack([x, y, z], axis=2)
            normal /= np.sum(normal**2.0, axis=2, keepdims=True)**0.5
            normal_image = (normal * 127.5 + 127.5).clip(0,
                                                         255).astype(np.uint8)

            return depth_image, normal_image


class HEDNetwork(torch.nn.Module):

    def __init__(self, model_path):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False))

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False))

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False))

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False))

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1), torch.nn.ReLU(inplace=False))

        self.netScoreOne = torch.nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(
            in_channels=128,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0)
        self.netScoreThr = torch.nn.Conv2d(
            in_channels=256,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0)
        self.netScoreFou = torch.nn.Conv2d(
            in_channels=512,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0)
        self.netScoreFiv = torch.nn.Conv2d(
            in_channels=512,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=5,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0), torch.nn.Sigmoid())

        self.load_state_dict({
            strKey.replace('module', 'net'): tenWeight
            for strKey, tenWeight in torch.load(model_path).items()
        })

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(
            data=[104.00698793, 116.66876762, 122.67891434],
            dtype=tenInput.dtype,
            device=tenInput.device).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(
            input=tenScoreOne,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(
            input=tenScoreTwo,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(
            input=tenScoreThr,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(
            input=tenScoreFou,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(
            input=tenScoreFiv,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode='bilinear',
            align_corners=False)

        return self.netCombine(
            torch.cat([
                tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv
            ], 1))


class CannyDetector:

    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)


class HEDdetector:

    def __init__(self, annotator_ckpts_path, device='cuda'):
        modelpath = os.path.join(annotator_ckpts_path, 'network-bsds500.pth')
        self.netNetwork = HEDNetwork(modelpath).to(device).eval()

    def __call__(self, input_image):
        assert input_image.ndim == 3
        input_image = input_image[:, :, ::-1].copy()
        with torch.no_grad():
            image_hed = torch.from_numpy(input_image).float().cuda()
            image_hed = image_hed / 255.0
            image_hed = rearrange(image_hed, 'h w c -> 1 c h w')
            edge = self.netNetwork(image_hed)[0]
            edge = (edge.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            return edge[0]


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True):
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    return mmcv.bgr2rgb(img)


class SegformerDetector:

    def __init__(self, annotator_ckpts_path, device='cuda'):
        modelpath = os.path.join(
            annotator_ckpts_path,
            'segformer_mit-b4_512x512_160k_ade20k_20220620_112216-4fa4f58f.pth'
        )
        config_file = os.path.join(
            annotator_ckpts_path.replace('ckpt/annotator/', ''),
            'config/config.py')
        self.model = init_segmentor(config_file, modelpath).to(device)

    def __call__(self, img):
        result = inference_segmentor(self.model, img)
        res_img = show_result_pyplot(
            self.model, img, result, get_palette('ade'), opacity=1)

        return res_img
