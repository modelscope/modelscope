import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if not ('DISPLAY' in os.environ):
    import matplotlib as mpl
    mpl.use('Agg')

cmap = plt.cm.jet


def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    return depth.astype('uint8')


def merge_into_row(ele, pred):

    def preprocess_depth(x):
        y = np.squeeze(x.data.cpu().numpy())
        return depth_colorize(y)

    # if is gray, transforms to rgb
    img_list = []
    if 'rgb' in ele:
        rgb = np.squeeze(ele['rgb'][0, ...].data.cpu().numpy())
        rgb = np.transpose(rgb, (1, 2, 0))
        img_list.append(rgb)
    elif 'g' in ele:
        g = np.squeeze(ele['g'][0, ...].data.cpu().numpy())
        g = np.array(Image.fromarray(g).convert('RGB'))
        img_list.append(g)
    if 'd' in ele:
        img_list.append(preprocess_depth(ele['d'][0, ...]))
    img_list.append(preprocess_depth(pred[0, ...]))
    if 'gt' in ele:
        img_list.append(preprocess_depth(ele['gt'][0, ...]))

    img_merge = np.hstack(img_list)
    return img_merge.astype('uint8')


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)


def save_depth_as_uint16png(img, filename):
    img = (img * 256).astype('uint16')
    cv2.imwrite(filename, img)


if ('DISPLAY' in os.environ):
    f, axarr = plt.subplots(4, 1)
    plt.tight_layout()
    plt.ion()


def display_warping(rgb_tgt, pred_tgt, warped):

    def preprocess(rgb_tgt, pred_tgt, warped):
        rgb_tgt = 255 * np.transpose(
            np.squeeze(rgb_tgt.data.cpu().numpy()), (1, 2, 0))  # H, W, C
        # depth = np.squeeze(depth.cpu().numpy())
        # depth = depth_colorize(depth)

        # convert to log-scale
        pred_tgt = np.squeeze(pred_tgt.data.cpu().numpy())
        # pred_tgt[pred_tgt<=0] = 0.9 # remove negative predictions
        # pred_tgt = np.log10(pred_tgt)

        pred_tgt = depth_colorize(pred_tgt)

        warped = 255 * np.transpose(
            np.squeeze(warped.data.cpu().numpy()), (1, 2, 0))  # H, W, C
        recon_err = np.absolute(
            warped.astype('float') - rgb_tgt.astype('float')) * (
                warped > 0)
        recon_err = recon_err[:, :, 0] + recon_err[:, :, 1] + recon_err[:, :,
                                                                        2]
        recon_err = depth_colorize(recon_err)
        return rgb_tgt.astype('uint8'), warped.astype(
            'uint8'), recon_err, pred_tgt

    rgb_tgt, warped, recon_err, pred_tgt = preprocess(rgb_tgt, pred_tgt,
                                                      warped)

    # 1st column
    # column = 0
    axarr[0].imshow(rgb_tgt)
    axarr[0].axis('off')
    axarr[0].axis('equal')
    # axarr[0, column].set_title('rgb_tgt')

    axarr[1].imshow(warped)
    axarr[1].axis('off')
    axarr[1].axis('equal')
    # axarr[1, column].set_title('warped')

    axarr[2].imshow(recon_err, 'hot')
    axarr[2].axis('off')
    axarr[2].axis('equal')
    # axarr[2, column].set_title('recon_err error')

    axarr[3].imshow(pred_tgt, 'hot')
    axarr[3].axis('off')
    axarr[3].axis('equal')
    # axarr[3, column].set_title('pred_tgt')

    # plt.show()
    plt.pause(0.001)
