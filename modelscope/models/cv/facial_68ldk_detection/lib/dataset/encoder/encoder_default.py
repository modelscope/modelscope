import copy
import numpy as np

import torch
import torch.nn.functional as F


class encoder_default:
    def __init__(self, image_height, image_width, scale=0.25, sigma=1.5):
        self.image_height = image_height
        self.image_width = image_width
        self.scale = scale
        self.sigma = sigma

    def generate_heatmap(self, points):
        # points = (num_pts, 2)
        h, w = self.image_height, self.image_width
        pointmaps = []
        for i in range(len(points)):
            pointmap = np.zeros([h, w], dtype=np.float32)
            # align_corners: False.
            point = copy.deepcopy(points[i])
            point[0] = max(0, min(w - 1, point[0]))
            point[1] = max(0, min(h - 1, point[1]))
            pointmap = self._circle(pointmap, point, sigma=self.sigma)

            pointmaps.append(pointmap)
        pointmaps = np.stack(pointmaps, axis=0) / 255.0
        pointmaps = torch.from_numpy(pointmaps).float().unsqueeze(0)
        pointmaps = F.interpolate(pointmaps, size=(int(w * self.scale), int(h * self.scale)), mode='bilinear',
                                  align_corners=False).squeeze()
        return pointmaps

    def _circle(self, img, pt, sigma=1.0, label_type='Gaussian'):
        # Check that any part of the gaussian is in-bounds
        tmp_size = sigma * 3
        ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
        br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
        if (ul[0] > img.shape[1] - 1 or ul[1] > img.shape[0] - 1 or
                br[0] - 1 < 0 or br[1] - 1 < 0):
            # If not, just return the image as is
            return img

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        if label_type == 'Gaussian':
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        else:
            g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = 255 * g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return img
