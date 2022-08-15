import cv2
import numpy as np


def numpy_to_cv2img(vis_img):
    """to convert a np.array Hotmap with shape(h, w) to cv2 img

    Args:
        vis_img (np.array): input data

    Returns:
        cv2 img
    """
    vis_img = (vis_img - vis_img.min()) / (
        vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    return vis_img
