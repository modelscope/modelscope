# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import os
import shutil
from typing import Any, Dict

import cv2
import face_alignment
import numpy as np
import PIL.Image
import tensorflow as tf
import torch
from scipy.io import loadmat, savemat

from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_reconstruction.models.facelandmark.large_base_lmks_infer import \
    LargeBaseLmkInfer
from modelscope.models.cv.face_reconstruction.utils import (
    POS, align_for_lm, draw_line, enlarged_bbox, extract_5p, image_warp_grid1,
    load_lm3d, mesh_to_string, read_obj, resize_n_crop_img,
    resize_on_long_side, spread_flow, write_obj)
from modelscope.models.cv.head_reconstruction.models.head_segmentation import \
    HeadSegmentor
from modelscope.models.cv.head_reconstruction.models.tex_processor import \
    TexProcesser
from modelscope.models.cv.skin_retouching.retinaface.predict_single import \
    Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import create_device, device_placement
from modelscope.utils.logger import get_logger

try:
    from torch.hub import get_dir
except BaseException:
    from torch.hub import _get_torch_home as get_dir

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()

logger = get_logger()


@PIPELINES.register_module(
    Tasks.head_reconstruction, module_name=Pipelines.head_reconstruction)
class HeadReconstructionPipeline(Pipeline):

    def __init__(self, model: str, device: str, hair_tex=False):
        """The inference pipeline for head reconstruction task.

        Args:
            model (`str` or `Model` or module instance): A model instance or a model local dir
                or a model id in the model hub.
            device ('str'): device str, should be either cpu, cuda, gpu, gpu:X or cuda:X.

        Example:
            >>> from modelscope.pipelines import pipeline
            >>> test_image = 'data/test/images/face_reconstruction.jpg'
            >>> pipeline_headRecon = pipeline('head-reconstruction',
                model='damo/cv_HRN_head-reconstruction')
            >>> result = pipeline_headRecon(test_image)
            >>> mesh = result[OutputKeys.OUTPUT]['mesh']
            >>> texture_map = result[OutputKeys.OUTPUT_IMG]
            >>> mesh['texture_map'] = texture_map
            >>> write_obj('head_reconstruction.obj', mesh)
        """
        super().__init__(model=model, device=device)

        model_root = model
        bfm_folder = os.path.join(model_root, 'assets')
        checkpoint_path = os.path.join(model_root, ModelFile.TORCH_MODEL_FILE)

        config_path = os.path.join(model_root, ModelFile.CONFIGURATION)
        logger.info(f'loading config from {config_path}')
        self.cfg = Config.from_file(config_path)

        self.hair_tex = hair_tex

        if 'gpu' in device:
            self.device_name_ = 'cuda'
        else:
            self.device_name_ = device
        self.device_name_ = self.device_name_.lower()
        lmks_cpkt_path = os.path.join(model_root, 'large_base_net.pth')
        self.large_base_lmks_model = LargeBaseLmkInfer.model_preload(
            lmks_cpkt_path, self.device_name_ == 'cuda')
        self.detector = Model(max_size=512, device=self.device_name_)
        detector_ckpt_name = 'retinaface_resnet50_2020-07-20_old_torch.pth'
        state_dict = torch.load(
            os.path.join(os.path.dirname(lmks_cpkt_path), detector_ckpt_name),
            map_location='cpu')
        self.detector.load_state_dict(state_dict)
        self.detector.eval()

        device = torch.device(self.device_name_)
        self.model.set_device(device)
        self.model.setup(checkpoint_path)
        self.model.parallelize()
        self.model.eval()
        self.model.set_render()

        hub_dir = get_dir()
        save_ckpt_dir = os.path.join(hub_dir, 'checkpoints')
        if not os.path.exists(save_ckpt_dir):
            os.makedirs(save_ckpt_dir)
        shutil.copy(
            os.path.join(model_root, 'face_alignment', 's3fd-619a316812.pth'),
            save_ckpt_dir)
        shutil.copy(
            os.path.join(model_root, 'face_alignment',
                         '3DFAN4-4a694010b9.zip'), save_ckpt_dir)
        shutil.copy(
            os.path.join(model_root, 'face_alignment', 'depth-6c4283c0e0.zip'),
            save_ckpt_dir)
        self.lm_sess = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.THREE_D,
            flip_input=False)  # face_alignment.LandmarksType._3D

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True
        g1 = tf.Graph()
        self.face_sess = tf.Session(graph=g1, config=config)
        with self.face_sess.as_default():
            with g1.as_default():
                with tf.gfile.FastGFile(
                        os.path.join(model_root, 'segment_face.pb'),
                        'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    self.face_sess.graph.as_default()
                    tf.import_graph_def(graph_def, name='')
                    self.face_sess.run(tf.global_variables_initializer())

        self.head_segmentor = HeadSegmentor(model_root=model_root)

        self.tex_processor = TexProcesser(model_root=model_root)

        self.lm3d_std = load_lm3d(bfm_folder)
        self.align_params = loadmat(
            '{}/assets/BBRegressorParam_r.mat'.format(model_root))

        device = create_device(self.device_name)
        self.device = device

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if isinstance(input, str):
            img = LoadImage.convert_to_ndarray(input)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = img.astype(float)
        else:
            img = input.astype(float)
        result = {'img': img}
        return result

    def align_img(self,
                  img,
                  lm,
                  lm3D,
                  mask=None,
                  target_size=224.,
                  rescale_factor=102.):
        """
        Return:
            transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
            img_new            --PIL.Image  (target_size, target_size, 3)
            lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
            mask_new           --PIL.Image  (target_size, target_size)

        Parameters:
            img                --PIL.Image  (raw_H, raw_W, 3)
            lm                 --numpy.array  (68, 2), y direction is opposite to v direction
            lm3D               --numpy.array  (5, 3)
            mask               --PIL.Image  (raw_H, raw_W, 3)
        """

        w0, h0 = img.size
        if lm.shape[0] != 5:
            lm5p = extract_5p(lm)
        else:
            lm5p = lm

        # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
        t, s = POS(lm5p.transpose(), lm3D.transpose())
        s = rescale_factor / s

        # processing the image
        img_new, lm_new, mask_new = resize_n_crop_img(
            img, lm, t, s, target_size=target_size, mask=mask)
        trans_params = np.array([w0, h0, s, t[0][0], t[1][0]])

        return trans_params, img_new, lm_new, mask_new

    def read_data(self,
                  img,
                  lm,
                  lm3d_std,
                  to_tensor=True,
                  image_res=1024,
                  img_fat=None,
                  head_mask=None,
                  rescale_factor=75.0):
        # to RGB
        im = PIL.Image.fromarray(img[..., ::-1])
        W, H = im.size
        lm[:, -1] = H - 1 - lm[:, -1]

        head_mask = PIL.Image.fromarray(head_mask)
        im_fat = PIL.Image.fromarray(img_fat[..., ::-1])

        _, im_lr_coeff, lm_lr_coeff, _ = self.align_img(im, lm, lm3d_std)
        _, im_lr, lm_lr, mask_lr_head = self.align_img(
            im, lm, lm3d_std, mask=head_mask, rescale_factor=rescale_factor)
        _, im_hd, lm_hd, _ = self.align_img(
            im_fat,
            lm,
            lm3d_std,
            target_size=image_res,
            rescale_factor=rescale_factor * image_res / 224)

        mask_lr = self.face_sess.run(
            self.face_sess.graph.get_tensor_by_name('output_alpha:0'),
            feed_dict={'input_image:0': np.array(im_lr)})

        if to_tensor:
            im_lr = torch.tensor(
                np.array(im_lr) / 255.,
                dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            im_hd = torch.tensor(
                np.array(im_hd) / 255.,
                dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            mask_lr = torch.tensor(
                np.array(mask_lr) / 255., dtype=torch.float32)[None,
                                                               None, :, :]
            mask_lr_head = torch.tensor(
                np.array(mask_lr_head) / 255., dtype=torch.float32)[
                    None, None, :, :] if mask_lr_head is not None else None
            lm_lr = torch.tensor(lm_lr).unsqueeze(0)
            lm_hd = torch.tensor(lm_hd).unsqueeze(0)
            im_lr_coeff = torch.tensor(
                np.array(im_lr_coeff) / 255.,
                dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            lm_lr_coeff = torch.tensor(lm_lr_coeff).unsqueeze(0)
        return im_lr, lm_lr, im_hd, lm_hd, mask_lr, mask_lr_head, im_lr_coeff, lm_lr_coeff

    def prepare_data(self, img, lm_sess, five_points=None):
        input_img, scale, bbox = align_for_lm(
            img, five_points,
            self.align_params)  # align for 68 landmark detection

        if scale == 0:
            return None

        # detect landmarks
        input_img = np.reshape(input_img, [1, 224, 224, 3]).astype(np.float32)

        input_img = input_img[0, :, :, ::-1]
        landmark = lm_sess.get_landmarks_from_image(input_img)[0]

        landmark = landmark[:, :2] / scale
        landmark[:, 0] = landmark[:, 0] + bbox[0]
        landmark[:, 1] = landmark[:, 1] + bbox[1]

        return landmark

    def infer_lmks(self, img_bgr):
        INPUT_SIZE = 224
        ENLARGE_RATIO = 1.35

        landmarks = []

        rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.predict_jsons(rgb_image)

        boxes = []
        for anno in results:
            if anno['score'] == -1:
                break
            boxes.append({
                'x1': anno['bbox'][0],
                'y1': anno['bbox'][1],
                'x2': anno['bbox'][2],
                'y2': anno['bbox'][3]
            })

        for detect_result in boxes:
            x1 = detect_result['x1']
            y1 = detect_result['y1']
            x2 = detect_result['x2']
            y2 = detect_result['y2']

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            cx = (x2 + x1) / 2
            cy = (y2 + y1) / 2

            sz = max(h, w) * ENLARGE_RATIO

            x1 = cx - sz / 2
            y1 = cy - sz / 2
            trans_x1 = x1
            trans_y1 = y1
            x2 = x1 + sz
            y2 = y1 + sz

            height, width, _ = rgb_image.shape
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            crop_img = rgb_image[int(y1):int(y2), int(x1):int(x2)]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                crop_img = cv2.copyMakeBorder(
                    crop_img,
                    int(dy),
                    int(edy),
                    int(dx),
                    int(edx),
                    cv2.BORDER_CONSTANT,
                    value=(103.94, 116.78, 123.68))
            crop_img = cv2.resize(crop_img, (INPUT_SIZE, INPUT_SIZE))

            base_lmks = LargeBaseLmkInfer.infer_img(
                crop_img, self.large_base_lmks_model,
                self.device_name_ == 'cuda')

            inv_scale = sz / INPUT_SIZE

            affine_base_lmks = np.zeros((106, 2))
            for idx in range(106):
                affine_base_lmks[idx][
                    0] = base_lmks[0][idx * 2 + 0] * inv_scale + trans_x1
                affine_base_lmks[idx][
                    1] = base_lmks[0][idx * 2 + 1] * inv_scale + trans_y1

            x1 = np.min(affine_base_lmks[:, 0])
            y1 = np.min(affine_base_lmks[:, 1])
            x2 = np.max(affine_base_lmks[:, 0])
            y2 = np.max(affine_base_lmks[:, 1])

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            cx = (x2 + x1) / 2
            cy = (y2 + y1) / 2

            sz = max(h, w) * ENLARGE_RATIO

            x1 = cx - sz / 2
            y1 = cy - sz / 2
            trans_x1 = x1
            trans_y1 = y1
            x2 = x1 + sz
            y2 = y1 + sz

            height, width, _ = rgb_image.shape
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            crop_img = rgb_image[int(y1):int(y2), int(x1):int(x2)]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                crop_img = cv2.copyMakeBorder(
                    crop_img,
                    int(dy),
                    int(edy),
                    int(dx),
                    int(edx),
                    cv2.BORDER_CONSTANT,
                    value=(103.94, 116.78, 123.68))
            crop_img = cv2.resize(crop_img, (INPUT_SIZE, INPUT_SIZE))

            base_lmks = LargeBaseLmkInfer.infer_img(
                crop_img, self.large_base_lmks_model,
                self.device_name_.lower() == 'cuda')

            inv_scale = sz / INPUT_SIZE

            affine_base_lmks = np.zeros((106, 2))
            for idx in range(106):
                affine_base_lmks[idx][
                    0] = base_lmks[0][idx * 2 + 0] * inv_scale + trans_x1
                affine_base_lmks[idx][
                    1] = base_lmks[0][idx * 2 + 1] * inv_scale + trans_y1

            landmarks.append(affine_base_lmks)

        return boxes, landmarks

    def find_face_contour(self, image):

        boxes, landmarks = self.infer_lmks(image)
        landmarks = np.array(landmarks)

        args = [[0, 33, False], [33, 38, False], [42, 47, False],
                [51, 55, False], [57, 64, False], [66, 74, True],
                [75, 83, True], [84, 96, True]]

        roi_bboxs = []

        for i in range(len(boxes)):
            roi_bbox = enlarged_bbox([
                boxes[i]['x1'], boxes[i]['y1'], boxes[i]['x2'], boxes[i]['y2']
            ], image.shape[1], image.shape[0], 0.5)
            roi_bbox = [int(x) for x in roi_bbox]
            roi_bboxs.append(roi_bbox)

        people_maps = []

        for i in range(landmarks.shape[0]):
            landmark = landmarks[i, :, :]
            maps = []
            whole_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

            roi_box = roi_bboxs[i]
            roi_box_width = roi_box[2] - roi_box[0]
            roi_box_height = roi_box[3] - roi_box[1]
            short_side_length = roi_box_width if roi_box_width < roi_box_height else roi_box_height

            line_width = short_side_length // 10

            if line_width == 0:
                line_width = 1

            kernel_size = line_width * 2
            gaussian_kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

            for t, arg in enumerate(args):
                mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
                draw_line(mask, landmark[arg[0]:arg[1]], (255, 255, 255),
                          line_width, arg[2])
                mask = cv2.GaussianBlur(mask,
                                        (gaussian_kernel, gaussian_kernel), 0)
                if t >= 1:
                    draw_line(whole_mask, landmark[arg[0]:arg[1]],
                              (255, 255, 255), line_width * 2, arg[2])
                maps.append(mask)
            whole_mask = cv2.GaussianBlur(whole_mask,
                                          (gaussian_kernel, gaussian_kernel),
                                          0)
            maps.append(whole_mask)
            people_maps.append(maps)

        return people_maps[0], boxes

    def fat_face(self, img, degree=0.04):

        _img, scale = resize_on_long_side(img, 800)

        contour_maps, boxes = self.find_face_contour(_img)

        contour_map = contour_maps[0]

        boxes = boxes[0]

        Flow = np.zeros(
            shape=(contour_map.shape[0], contour_map.shape[1], 2),
            dtype=np.float32)

        box_center = [(boxes['x1'] + boxes['x2']) / 2,
                      (boxes['y1'] + boxes['y2']) / 2]

        box_length = max(
            abs(boxes['y1'] - boxes['y2']), abs(boxes['x1'] - boxes['x2']))

        value_1 = 2 * (Flow.shape[0] - box_center[1] - 1)
        value_2 = 2 * (Flow.shape[1] - box_center[0] - 1)
        value_list = [
            box_length * 2, 2 * (box_center[0] - 1), 2 * (box_center[1] - 1),
            value_1, value_2
        ]
        flow_box_length = min(value_list)
        flow_box_length = int(flow_box_length)

        sf = spread_flow(100, flow_box_length * degree)
        sf = cv2.resize(sf, (flow_box_length, flow_box_length))

        Flow[int(box_center[1]
                 - flow_box_length / 2):int(box_center[1]
                                            + flow_box_length / 2),
             int(box_center[0]
                 - flow_box_length / 2):int(box_center[0]
                                            + flow_box_length / 2)] = sf

        Flow = Flow * np.dstack((contour_map, contour_map)) / 255.0

        inter_face_maps = contour_maps[-1]

        Flow = Flow * (1.0 - np.dstack(
            (inter_face_maps, inter_face_maps)) / 255.0)

        Flow = cv2.resize(Flow, (img.shape[1], img.shape[0]))

        Flow = Flow / scale

        pred, top_bound, bottom_bound, left_bound, right_bound = image_warp_grid1(
            Flow[..., 0], Flow[..., 1], img, 1.0, [0, 0, 0, 0])

        return pred

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        rgb_image = input['img'].cpu().numpy().astype(np.uint8)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        img = bgr_image

        if img.shape[0] > 2000 or img.shape[1] > 2000:
            img, _ = resize_on_long_side(img, 1500)

        box, results = self.infer_lmks(img)

        if results is None or np.array(results).shape[0] == 0:
            return {}

        fatbgr = self.fat_face(img)

        landmarks = []
        results = results[0]
        for idx in [74, 83, 54, 84, 90]:
            landmarks.append([results[idx][0], results[idx][1]])
        landmarks = np.array(landmarks)

        landmarks = self.prepare_data(img, self.lm_sess, five_points=landmarks)

        head_mask = self.head_segmentor.process(img)[0]

        im_tensor, lm_tensor, im_hd_tensor, lm_hd_tensor, mask, head_mask, im_co, lm_co = self.read_data(
            img, landmarks, self.lm3d_std, img_fat=fatbgr, head_mask=head_mask)

        data = {
            'imgs': im_tensor,
            'imgs_hd': im_hd_tensor,
            'lms': lm_tensor,
            'lms_hd': lm_hd_tensor,
            'face_mask': mask,
            'head_mask': head_mask,
            'imgs_coeff': im_co,
            'lms_coeff': lm_co,
        }
        self.model.set_input(data)  # unpack data from data loader

        output = self.model()  # run inference

        assert output is not None

        tex_map = output['tex_map'].astype(np.float32)

        # post-process texture map
        tex_map = self.tex_processor.post_process_texture(
            tex_map, hair_tex=self.hair_tex)

        head_mesh = {
            'vertices': output['vertices'],
            'faces': output['triangles'] + 1,
            'UVs': output['uvs'],
            'faces_uv': output['faces_uv'],
            'normals': output['normals'],
            'texture_map': tex_map
        }

        results = {
            'mesh': head_mesh,
        }

        return {
            OutputKeys.OUTPUT_OBJ: None,
            OutputKeys.OUTPUT_IMG: tex_map,
            OutputKeys.OUTPUT: results
        }

    def postprocess(self, inputs, **kwargs) -> Dict[str, Any]:
        render = kwargs.get('render', False)
        output_obj = inputs[OutputKeys.OUTPUT_OBJ]
        texture_map = inputs[OutputKeys.OUTPUT_IMG]
        results = inputs[OutputKeys.OUTPUT]

        if render:
            output_obj = io.BytesIO()
            mesh_str = mesh_to_string(results['mesh'])
            mesh_bytes = mesh_str.encode(encoding='utf-8')
            output_obj.write(mesh_bytes)

        result = {
            OutputKeys.OUTPUT_OBJ: output_obj,
            OutputKeys.OUTPUT_IMG: texture_map,
            OutputKeys.OUTPUT: None if render else results,
        }
        return result
