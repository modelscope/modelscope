# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import cv2
import numpy as np


class YOLOXONNX(object):
    """
    Product detection model with onnx inference
    """

    def __init__(self, onnx_path, multi_detect=False):
        """Create product detection model
        Args:
             onnx_path: onnx model path for product detection
             multi_detect: detection parameter, should be set as False

        """
        self.input_reso = 416
        self.iou_thr = 0.45
        self.score_thr = 0.3
        self.img_shape = tuple([self.input_reso, self.input_reso, 3])
        self.num_classes = 13
        self.onnx_path = onnx_path
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.ort_session = ort.InferenceSession(
            self.onnx_path,
            sess_options=options,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.with_p6 = False
        self.multi_detect = multi_detect

    def format_judge(self, img):
        m_min_width = 100
        m_min_height = 100

        height, width, c = img.shape

        if width * height > 1024 * 1024:
            if height > width:
                long_side = height
                short_side = width
                long_ratio = float(long_side) / 1024.0
                short_ratio = float(short_side) / float(m_min_width)
            else:
                long_side = width
                short_side = height
                long_ratio = float(long_side) / 1024.0
                short_ratio = float(short_side) / float(m_min_height)

            if long_side == height:
                if long_ratio < short_ratio:
                    height_new = 1024
                    width_new = (int)((1024 * width) / height)

                    img_res = cv2.resize(img, (width_new, height_new),
                                         cv2.INTER_LINEAR)
                else:
                    height_new = (int)((m_min_width * height) / width)
                    width_new = m_min_width

                    img_res = cv2.resize(img, (width_new, height_new),
                                         cv2.INTER_LINEAR)

            elif long_side == width:
                if long_ratio < short_ratio:
                    height_new = (int)((1024 * height) / width)
                    width_new = 1024

                    img_res = cv2.resize(img, (width_new, height_new),
                                         cv2.INTER_LINEAR)
                else:
                    width_new = (int)((m_min_height * width) / height)
                    height_new = m_min_height

                    img_res = cv2.resize(img, (width_new, height_new),
                                         cv2.INTER_LINEAR)
        else:
            img_res = img

        return img_res

    def preprocess(self, image, input_size, swap=(2, 0, 1)):
        """
        Args:
            image, cv2 image with BGR format
            input_size, model input size
        """
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[:int(img.shape[0] * r), :int(img.shape[1]
                                                * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def cal_iou(self, val1, val2):
        x11, y11, x12, y12 = val1
        x21, y21, x22, y22 = val2

        leftX = max(x11, x21)
        topY = max(y11, y21)
        rightX = min(x12, x22)
        bottomY = min(y12, y22)
        if rightX < leftX or bottomY < topY:
            return 0
        area = float((rightX - leftX) * (bottomY - topY))
        barea = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - area
        if barea <= 0:
            return 0
        return area / barea

    def nms(self, boxes, scores, nms_thr):
        """
            Single class NMS implemented in Numpy.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep

    def multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        """
            Multiclass NMS implemented in Numpy
        """
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate([
                        valid_boxes[keep], valid_scores[keep, None], cls_inds
                    ], 1)
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)

    def postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs

    def get_new_box_order(self, bboxes, labels, img_h, img_w):
        """
            refine bbox score
        """
        bboxes = np.hstack((bboxes, np.zeros((bboxes.shape[0], 1))))
        scores = bboxes[:, 4]
        order = scores.argsort()[::-1]
        bboxes_temp = bboxes[order]
        labels_temp = labels[order]
        bboxes = np.empty((0, 6))
        # import pdb;pdb.set_trace()
        bboxes = np.vstack((bboxes, bboxes_temp[0].tolist()))
        labels = np.empty((0, ))

        labels = np.hstack((labels, [labels_temp[0]]))
        for i in range(1, bboxes_temp.shape[0]):
            iou_max = 0
            for j in range(bboxes.shape[0]):
                iou_temp = self.cal_iou(bboxes_temp[i][:4], bboxes[j][:4])
                if (iou_temp > iou_max):
                    iou_max = iou_temp
            if (iou_max < 0.45):
                bboxes = np.vstack((bboxes, bboxes_temp[i].tolist()))
                labels = np.hstack((labels, [labels_temp[i]]))

        num_03 = scores > 0.3
        num_03 = num_03.sum()
        num_out = max(num_03, 1)
        bboxes = bboxes[:num_out, :]
        labels = labels[:num_out]

        return bboxes, labels

    def forward(self, img_input, cid='0', sub_class=False):
        """
        forward for product detection
        """
        input_shape = self.img_shape

        img, ratio = self.preprocess(img_input, input_shape)
        img_h, img_w = img_input.shape[:2]

        ort_inputs = {
            self.ort_session.get_inputs()[0].name: img[None, :, :, :]
        }

        output = self.ort_session.run(None, ort_inputs)

        predictions = self.postprocess(output[0], input_shape, self.with_p6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = self.multiclass_nms(
            boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

        if dets is None:
            top1_bbox_str = str(0) + ',' + str(img_w) + ',' + str(
                0) + ',' + str(img_h)
            crop_img = img_input.copy()
            coord = top1_bbox_str
        else:
            bboxes = dets[:, :5]
            labels = dets[:, 5]

            if not self.multi_detect:
                cid = int(cid)
                if (not sub_class):
                    if cid > -1:
                        if cid == 0:  # cloth
                            cid_ind1 = np.where(labels < 3)
                            cid_ind2 = np.where(labels == 9)
                            cid_ind = np.hstack((cid_ind1[0], cid_ind2[0]))
                            scores = bboxes[cid_ind, -1]  # 0, 1, 2, 9

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                        elif cid == 3:  # bag
                            cid_ind = np.where(labels == 3)
                            scores = bboxes[cid_ind, -1]  # 3

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                        elif cid == 4:  # shoe
                            cid_ind = np.where(labels == 4)
                            scores = bboxes[cid_ind, -1]  # 4

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                        else:  # other
                            cid_ind5 = np.where(labels == 5)
                            cid_ind6 = np.where(labels == 6)
                            cid_ind7 = np.where(labels == 7)
                            cid_ind8 = np.where(labels == 8)
                            cid_ind10 = np.where(labels == 10)
                            cid_ind11 = np.where(labels == 11)
                            cid_ind12 = np.where(labels == 12)
                            cid_ind = np.hstack(
                                (cid_ind5[0], cid_ind6[0], cid_ind7[0],
                                 cid_ind8[0], cid_ind10[0], cid_ind11[0],
                                 cid_ind12[0]))
                            scores = bboxes[cid_ind, -1]  # 5,6,7,8,10,11,12

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                    else:
                        bboxes, labels = self.get_new_box_order(
                            bboxes, labels, img_h, img_w)
                else:
                    if cid > -1:
                        if cid == 0:  # upper
                            cid_ind = np.where(labels == 0)

                            scores = bboxes[cid_ind, -1]  # 0, 1, 2, 9

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                        elif cid == 1:  # skirt
                            cid_ind = np.where(labels == 1)
                            scores = bboxes[cid_ind, -1]  # 3

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                        elif cid == 2:  # lower
                            cid_ind = np.where(labels == 2)
                            scores = bboxes[cid_ind, -1]  # 3

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                        elif cid == 3:  # bag
                            cid_ind = np.where(labels == 3)
                            scores = bboxes[cid_ind, -1]  # 3

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                        elif cid == 4:  # shoe
                            cid_ind = np.where(labels == 4)
                            scores = bboxes[cid_ind, -1]  # 4

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                        elif cid == 5:  # access
                            cid_ind = np.where(labels == 5)
                            scores = bboxes[cid_ind, -1]  # 3

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                        elif cid == 7:  # beauty
                            cid_ind = np.where(labels == 6)
                            scores = bboxes[cid_ind, -1]  # 3

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                        elif cid == 9:  # furniture
                            cid_ind = np.where(labels == 8)
                            scores = bboxes[cid_ind, -1]  # 3

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                        elif cid == 21:  # underwear
                            cid_ind = np.where(labels == 9)
                            scores = bboxes[cid_ind, -1]  # 3

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)
                        elif cid == 22:  # digital
                            cid_ind = np.where(labels == 11)
                            scores = bboxes[cid_ind, -1]  # 3

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                        else:  # other
                            cid_ind5 = np.where(labels == 7)  # bottle
                            cid_ind6 = np.where(labels == 10)  # toy
                            cid_ind7 = np.where(labels == 12)  # toy
                            cid_ind = np.hstack(
                                (cid_ind5[0], cid_ind6[0], cid_ind7[0]))
                            scores = bboxes[cid_ind, -1]  # 5,6,7

                            if scores.size > 0:

                                bboxes = bboxes[cid_ind]
                                labels = labels[cid_ind]
                            bboxes, labels = self.get_new_box_order(
                                bboxes, labels, img_h, img_w)

                    else:
                        bboxes, labels = self.get_new_box_order(
                            bboxes, labels, img_h, img_w)
            else:
                bboxes, labels = self.get_new_box_order(
                    bboxes, labels, img_h, img_w)
            top1_bbox = bboxes[0].astype(np.int32)
            top1_bbox[0] = min(max(0, top1_bbox[0]), img_input.shape[1] - 1)
            top1_bbox[1] = min(max(0, top1_bbox[1]), img_input.shape[0] - 1)
            top1_bbox[2] = max(min(img_input.shape[1] - 1, top1_bbox[2]), 0)
            top1_bbox[3] = max(min(img_input.shape[0] - 1, top1_bbox[3]), 0)
            if not self.multi_detect:

                top1_bbox_str = str(top1_bbox[0]) + ',' + str(
                    top1_bbox[2]) + ',' + str(top1_bbox[1]) + ',' + str(
                        top1_bbox[3])  # x1, x2, y1, y2
                crop_img = img_input[top1_bbox[1]:top1_bbox[3],
                                     top1_bbox[0]:top1_bbox[2], :]
                coord = top1_bbox_str
                coord = ''
                for i in range(0, len(bboxes)):
                    top_bbox = bboxes[i].astype(np.int32)
                    top_bbox[0] = min(
                        max(0, top_bbox[0]), img_input.shape[1] - 1)
                    top_bbox[1] = min(
                        max(0, top_bbox[1]), img_input.shape[0] - 1)
                    top_bbox[2] = max(
                        min(img_input.shape[1] - 1, top_bbox[2]), 0)
                    top_bbox[3] = max(
                        min(img_input.shape[0] - 1, top_bbox[3]), 0)
                    coord = coord + str(top_bbox[0]) + ',' + str(
                        top_bbox[2]) + ',' + str(top_bbox[1]) + ',' + str(
                            top_bbox[3]) + ',' + str(bboxes[i][4]) + ',' + str(
                                bboxes[i][5]) + ';'

            else:
                coord = ''
                for i in range(0, len(bboxes)):
                    top_bbox = bboxes[i].astype(np.int32)
                    top_bbox[0] = min(
                        max(0, top_bbox[0]), img_input.shape[1] - 1)
                    top_bbox[1] = min(
                        max(0, top_bbox[1]), img_input.shape[0] - 1)
                    top_bbox[2] = max(
                        min(img_input.shape[1] - 1, top_bbox[2]), 0)
                    top_bbox[3] = max(
                        min(img_input.shape[0] - 1, top_bbox[3]), 0)
                    coord = coord + str(top_bbox[0]) + ',' + str(
                        top_bbox[2]) + ',' + str(top_bbox[1]) + ',' + str(
                            top_bbox[3]) + ',' + str(bboxes[i][4]) + ',' + str(
                                bboxes[i][5]) + ';'  # x1, x2, y1, y2, conf
                crop_img = img_input[top1_bbox[1]:top1_bbox[3],
                                     top1_bbox[0]:top1_bbox[2], :]

        crop_img = cv2.resize(crop_img, (224, 224))

        return coord, crop_img  # return top1 image and coord
