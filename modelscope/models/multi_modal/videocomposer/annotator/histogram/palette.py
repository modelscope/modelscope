r"""Modified from ``https://github.com/sergeyk/rayleigh''.
"""
import os
import os.path as osp
import numpy as np
from skimage.color import hsv2rgb, rgb2lab, lab2rgb
from skimage.io import imsave
from sklearn.metrics import euclidean_distances

__all__ = ['Palette']

def rgb2hex(rgb):
    return '#%02x%02x%02x' % tuple([int(round(255.0 * u)) for u in rgb])

def hex2rgb(hex):
    rgb = hex.strip('#')
    fn = lambda u: round(int(u, 16) / 255.0, 5)
    return fn(rgb[:2]), fn(rgb[2:4]), fn(rgb[4:6])

class Palette(object):
    r"""Create a color palette (codebook) in the form of a 2D grid of colors.
        Further, the rightmost column has num_hues gradations from black to white.

    Parameters:
        num_hues: number of colors with full lightness and saturation, in the middle.
        num_sat: number of rows above middle row that show the same hues with decreasing saturation.
    """
    def __init__(self, num_hues=11, num_sat=5, num_light=4):
        n = num_sat + 2 * num_light

        # hues
        if num_hues == 8:
            hues = np.tile(np.array([0.,  0.10,  0.15,  0.28, 0.51, 0.58, 0.77,  0.85]), (n, 1))
        elif num_hues == 9:
            hues = np.tile(np.array([0.,  0.10,  0.15,  0.28, 0.49, 0.54, 0.60, 0.7, 0.87]), (n, 1))
        elif num_hues == 10:
            hues = np.tile(np.array([0.,  0.10,  0.15,  0.28, 0.49, 0.54, 0.60, 0.66, 0.76, 0.87]), (n, 1))
        elif num_hues == 11:
            hues = np.tile(np.array([0.0, 0.0833, 0.166, 0.25, 0.333, 0.5, 0.56333, 0.666, 0.73, 0.803, 0.916]), (n, 1))
        else:
            hues = np.tile(np.linspace(0, 1, num_hues + 1)[:-1], (n, 1))
        
        # saturations
        sats = np.hstack((
            np.linspace(0, 1, num_sat + 2)[1:-1],
            1,
            [1] * num_light,
            [0.4] * (num_light - 1)))
        sats = np.tile(np.atleast_2d(sats).T, (1, num_hues))

        # lights
        lights = np.hstack((
            [1] * num_sat,
            1,
            np.linspace(1, 0.2, num_light + 2)[1:-1],
            np.linspace(1, 0.2, num_light + 2)[1:-2]))
        lights = np.tile(np.atleast_2d(lights).T, (1, num_hues))

        # colors
        rgb = hsv2rgb(np.dstack([hues, sats, lights]))
        gray = np.tile(np.linspace(1, 0, n)[:, np.newaxis, np.newaxis], (1, 1, 3))
        self.thumbnail = np.hstack([rgb, gray])

        # flatten
        rgb = rgb.T.reshape(3, -1).T
        gray = gray.T.reshape(3, -1).T
        self.rgb = np.vstack((rgb, gray))
        self.lab = rgb2lab(self.rgb[np.newaxis, :, :]).squeeze()
        self.hex = [rgb2hex(u) for u in self.rgb]
        self.lab_dists = euclidean_distances(self.lab, squared=True)
    
    def histogram(self, rgb_img, sigma=20):
        # compute histogram
        lab = rgb2lab(rgb_img).reshape((-1, 3))
        min_ind = np.argmin(euclidean_distances(lab, self.lab, squared=True), axis=1)
        hist = 1.0 * np.bincount(min_ind, minlength=self.lab.shape[0]) / lab.shape[0]

        # smooth histogram
        if sigma > 0:
            weight = np.exp(-self.lab_dists / (2.0 * sigma ** 2))
            weight = weight / weight.sum(1)[:, np.newaxis]
            hist = (weight * hist).sum(1)
            hist[hist < 1e-5] = 0
        return hist
    
    def get_palette_image(self, hist, percentile=90, width=200, height=50):
        # curate histogram
        ind = np.argsort(-hist)
        ind = ind[hist[ind] > np.percentile(hist, percentile)]
        hist = hist[ind] / hist[ind].sum()

        # draw palette
        nums = np.array(hist * width, dtype=int)
        array = np.vstack([np.tile(np.array(u), (v, 1)) for u, v in zip(self.rgb[ind], nums)])
        array = np.tile(array[np.newaxis, :, :], (height, 1, 1))
        if array.shape[1] < width:
            array = np.concatenate([array, np.zeros((height, width - array.shape[1], 3))], axis=1)
        return array

    def quantize_image(self, rgb_img):
        lab = rgb2lab(rgb_img).reshape((-1, 3))
        min_ind = np.argmin(euclidean_distances(lab, self.lab, squared=True), axis=1)
        quantized_lab = self.lab[min_ind]
        img = lab2rgb(quantized_lab.reshape(rgb_img.shape))
        return img
    
    def export(self, dirname):
        if not osp.exists(dirname):
            os.makedirs(dirname)
        
        # save thumbnail
        imsave(osp.join(dirname, 'palette.png'), self.thumbnail)

        # save html
        with open(osp.join(dirname, 'palette.html'), 'w') as f:
            html = '''
            <style>
                span {
                    width: 20px;
                    height: 20px;
                    margin: 2px;
                    padding: 0px;
                    display: inline-block;
                }
            </style>
            '''
            for row in self.thumbnail:
                for col in row:
                    html += '<a id="{0}"><span style="background-color: {0}" /></a>\n'.format(rgb2hex(col))
                html += '<br />\n'
            f.write(html)
