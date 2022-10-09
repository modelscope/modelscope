# The implementation is adopted from OSTrack,
# made publicly available under the MIT License at https://github.com/botaoye/OSTrack/
import torch

from modelscope.models.cv.video_single_object_tracking.config.ostrack import \
    cfg
from modelscope.models.cv.video_single_object_tracking.models.ostrack.ostrack import \
    build_ostrack
from modelscope.models.cv.video_single_object_tracking.utils.utils import (
    Preprocessor, clip_box, generate_mask_cond, hann2d, sample_target,
    transform_image_to_crop)


class OSTrack():

    def __init__(self, ckpt_path, device):
        network = build_ostrack(cfg)
        network.load_state_dict(
            torch.load(ckpt_path, map_location='cpu')['net'], strict=True)
        self.cfg = cfg
        if device.type == 'cuda':
            self.network = network.to(device)
        else:
            self.network = network
        self.network.eval()
        self.preprocessor = Preprocessor(device)
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        if device.type == 'cuda':
            self.output_window = hann2d(
                torch.tensor([self.feat_sz, self.feat_sz]).long(),
                centered=True).to(device)
        else:
            self.output_window = hann2d(
                torch.tensor([self.feat_sz, self.feat_sz]).long(),
                centered=True)
        self.frame_id = 0
        # for save boxes from all queries
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(
            image,
            info['init_bbox'],
            self.cfg.TEST.TEMPLATE_FACTOR,
            output_sz=self.cfg.TEST.TEMPLATE_SIZE)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(
                info['init_bbox'], resize_factor,
                template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1,
                                                 template.tensors.device,
                                                 template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(
            image,
            self.state,
            self.cfg.TEST.SEARCH_FACTOR,
            output_sz=self.cfg.TEST.SEARCH_SIZE)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors,
                search=x_dict.tensors,
                ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response,
                                                    out_dict['size_map'],
                                                    out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.cfg.TEST.SEARCH_SIZE
                    / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(
            self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        x1, y1, w, h = self.state
        x2 = x1 + w
        y2 = y1 + h
        return {'target_bbox': [x1, y1, x2, y2]}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[
            1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.cfg.TEST.SEARCH_SIZE / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def transform_bbox_to_crop(self,
                               box_in,
                               resize_factor,
                               device,
                               box_extract=None,
                               crop_type='template'):
        if crop_type == 'template':
            crop_sz = torch.Tensor(
                [self.cfg.TEST.TEMPLATE_SIZE, self.cfg.TEST.TEMPLATE_SIZE])
        elif crop_type == 'search':
            crop_sz = torch.Tensor(
                [self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE])
        else:
            raise NotImplementedError

        box_in = torch.tensor(box_in)
        if box_extract is None:
            box_extract = box_in
        else:
            box_extract = torch.tensor(box_extract)
        template_bbox = transform_image_to_crop(
            box_in, box_extract, resize_factor, crop_sz, normalize=True)
        template_bbox = template_bbox.view(1, 1, 4).to(device)

        return template_bbox
