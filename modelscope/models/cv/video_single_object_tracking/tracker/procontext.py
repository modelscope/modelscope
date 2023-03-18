# The ProContEXT implementation is also open-sourced by the authors,
# and available at https://github.com/jp-lan/ProContEXT
from copy import deepcopy

import torch

from modelscope.models.cv.video_single_object_tracking.models.procontext.procontext import \
    build_procontext
from modelscope.models.cv.video_single_object_tracking.utils.utils import (
    Preprocessor, clip_box, generate_mask_cond, hann2d, sample_target,
    transform_image_to_crop)


class ProContEXT():

    def __init__(self, ckpt_path, device, cfg):
        network = build_procontext(cfg)
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
        self.z_dict_list = []
        self.update_intervals = [100]

    def initialize(self, image, info: dict):
        # crop templates
        crop_resize_patches = [
            sample_target(
                image,
                info['init_bbox'],
                factor,
                output_sz=self.cfg.TEST.TEMPLATE_SIZE)
            for factor in self.cfg.TEST.TEMPLATE_FACTOR
        ]
        z_patch_arr, resize_factor, z_amask_arr = zip(*crop_resize_patches)
        for idx in range(len(z_patch_arr)):
            template = self.preprocessor.process(z_patch_arr[idx],
                                                 z_amask_arr[idx])
            with torch.no_grad():
                self.z_dict1 = template
            self.z_dict_list.append(self.z_dict1)
        self.box_mask_z = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            for i in range(len(self.cfg.TEST.TEMPLATE_FACTOR) * 2):
                template_bbox = self.transform_bbox_to_crop(
                    info['init_bbox'], resize_factor[0],
                    template.tensors.device).squeeze(1)
                self.box_mask_z.append(
                    generate_mask_cond(self.cfg, 1, template.tensors.device,
                                       template_bbox))

        # init dynamic templates with static templates
        for idx in range(len(self.cfg.TEST.TEMPLATE_FACTOR)):
            self.z_dict_list.append(deepcopy(self.z_dict_list[idx]))

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
            if isinstance(self.z_dict_list, (list, tuple)):
                self.z_dict = []
                for i in range(len(self.cfg.TEST.TEMPLATE_FACTOR) * 2):
                    self.z_dict.append(self.z_dict_list[i].tensors)
            out_dict = self.network.forward(
                template=self.z_dict,
                search=x_dict.tensors,
                ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        conf_score = out_dict['score']
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

        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % update_i == 0 and conf_score > 0.7:
                crop_resize_patches2 = [
                    sample_target(
                        image,
                        self.state,
                        factor,
                        output_sz=self.cfg.TEST.TEMPLATE_SIZE)
                    for factor in self.cfg.TEST.TEMPLATE_FACTOR
                ]
                z_patch_arr2, _, z_amask_arr2 = zip(*crop_resize_patches2)
                for idx_s in range(len(z_patch_arr2)):
                    template_t = self.preprocessor.process(
                        z_patch_arr2[idx_s], z_amask_arr2[idx_s])
                    self.z_dict_list[
                        idx_s
                        + len(self.cfg.TEST.TEMPLATE_FACTOR)] = template_t

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
