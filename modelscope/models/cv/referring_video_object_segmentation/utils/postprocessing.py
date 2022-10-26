# The implementation is adopted from MTTR,
# made publicly available under the Apache 2.0 License at https://github.com/mttr2021/MTTR

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class A2DSentencesPostProcess(nn.Module):
    """
    This module converts the model's output into the format expected by the coco api for the given task
    """

    def __init__(self):
        super(A2DSentencesPostProcess, self).__init__()

    @torch.inference_mode()
    def forward(self, outputs, resized_padded_sample_size,
                resized_sample_sizes, orig_sample_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            resized_padded_sample_size: size of samples (input to model) after size augmentation + padding.
            resized_sample_sizes: size of samples after size augmentation but without padding.
            orig_sample_sizes: original size of the samples (no augmentations or padding)
        """
        pred_is_referred = outputs['pred_is_referred']
        prob = F.softmax(pred_is_referred, dim=-1)
        scores = prob[..., 0]
        pred_masks = outputs['pred_masks']
        pred_masks = F.interpolate(
            pred_masks,
            size=resized_padded_sample_size,
            mode='bilinear',
            align_corners=False)
        pred_masks = (pred_masks.sigmoid() > 0.5)
        processed_pred_masks, rle_masks = [], []
        for f_pred_masks, resized_size, orig_size in zip(
                pred_masks, resized_sample_sizes, orig_sample_sizes):
            f_mask_h, f_mask_w = resized_size  # resized shape without padding
            # remove the samples' padding
            f_pred_masks_no_pad = f_pred_masks[:, :f_mask_h, :
                                               f_mask_w].unsqueeze(1)
            # resize the samples back to their original dataset (target) size for evaluation
            f_pred_masks_processed = F.interpolate(
                f_pred_masks_no_pad.float(), size=orig_size, mode='nearest')
            f_pred_rle_masks = [
                mask_util.encode(
                    np.array(
                        mask[0, :, :, np.newaxis], dtype=np.uint8,
                        order='F'))[0]
                for mask in f_pred_masks_processed.cpu()
            ]
            processed_pred_masks.append(f_pred_masks_processed)
            rle_masks.append(f_pred_rle_masks)
        predictions = [{
            'scores': s,
            'masks': m,
            'rle_masks': rle
        } for s, m, rle in zip(scores, processed_pred_masks, rle_masks)]
        return predictions


class ReferYoutubeVOSPostProcess(nn.Module):
    """
    This module converts the model's output into the format expected by the coco api for the given task
    """

    def __init__(self):
        super(ReferYoutubeVOSPostProcess, self).__init__()

    @torch.inference_mode()
    def forward(self, outputs, videos_metadata, samples_shape_with_padding):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            videos_metadata: a dictionary with each video's metadata.
            samples_shape_with_padding: size of the batch frames with padding.
        """
        pred_is_referred = outputs['pred_is_referred']
        prob_is_referred = F.softmax(pred_is_referred, dim=-1)
        # note we average on the temporal dim to compute score per trajectory:
        trajectory_scores = prob_is_referred[..., 0].mean(dim=0)
        pred_trajectory_indices = torch.argmax(trajectory_scores, dim=-1)
        pred_masks = rearrange(outputs['pred_masks'],
                               't b nq h w -> b t nq h w')
        # keep only the masks of the chosen trajectories:
        b = pred_masks.shape[0]
        pred_masks = pred_masks[torch.arange(b), :, pred_trajectory_indices]
        # resize the predicted masks to the size of the model input (which might include padding)
        pred_masks = F.interpolate(
            pred_masks,
            size=samples_shape_with_padding,
            mode='bilinear',
            align_corners=False)
        # apply a threshold to create binary masks:
        pred_masks = (pred_masks.sigmoid() > 0.5)
        # remove the padding per video (as videos might have different resolutions and thus different padding):
        preds_by_video = []
        for video_pred_masks, video_metadata in zip(pred_masks,
                                                    videos_metadata):
            # size of the model input batch frames without padding:
            resized_h, resized_w = video_metadata['resized_frame_size']
            video_pred_masks = video_pred_masks[:, :resized_h, :
                                                resized_w].unsqueeze(
                                                    1)  # remove the padding
            # resize the masks back to their original frames dataset size for evaluation:
            original_frames_size = video_metadata['original_frame_size']
            tuple_size = tuple(original_frames_size.cpu().numpy())
            video_pred_masks = F.interpolate(
                video_pred_masks.float(), size=tuple_size, mode='nearest')
            video_pred_masks = video_pred_masks.to(torch.uint8).cpu()
            # combine the predicted masks and the video metadata to create a final predictions dict:
            video_pred = {**video_metadata, **{'pred_masks': video_pred_masks}}
            preds_by_video.append(video_pred)
        return preds_by_video
