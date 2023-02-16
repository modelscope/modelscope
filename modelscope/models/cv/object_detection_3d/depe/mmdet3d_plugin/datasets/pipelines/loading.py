"""
The implementation here is modified based on PETR, originally Apache-2.0 license and publicly avaialbe at
https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/datasets/pipelines
"""
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        sweeps_num=5,
        to_float32=False,
        file_client_args=dict(backend='disk'),
        pad_empty_sweeps=False,
        sweep_range=[3, 27],
        sweeps_id=None,
        color_type='unchanged',
        sensors=[
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ],
        test_mode=True,
        prob=1.0,
    ):

        self.sweeps_num = sweeps_num
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = results['img']
        img_timestamp = results['img_timestamp']
        lidar_timestamp = results['timestamp']
        img_timestamp = [
            lidar_timestamp - timestamp for timestamp in img_timestamp
        ]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                results['pts_filename'] += [results['pts_filename'][0]]
                mean_time = (self.sweep_range[0]
                             + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend(
                    [time + mean_time for time in img_timestamp])
                for j in range(nums):
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(
                        np.copy(results['lidar2img'][j]))
                    results['intrinsics'].append(
                        np.copy(results['intrinsics'][j]))
                    results['extrinsics'].append(
                        np.copy(results['extrinsics'][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(results['sweeps']) <= self.sweeps_num:
                bin_size = len(results['sweeps']) / (self.sweeps_num + 1)
                choices = [
                    int(np.floor((i + 1) * bin_size))
                    for i in range(self.sweeps_num)
                ]
            elif self.test_mode:
                if self.sweep_range[1] <= len(results['sweeps']):
                    sweep_range = list(
                        range(self.sweep_range[0], self.sweep_range[1]))
                elif self.sweep_range[0] >= len(results['sweeps']):
                    sweep_range = list(range(0, len(results['sweeps'])))
                else:
                    sweep_range = list(
                        range(self.sweep_range[0], len(results['sweeps'])))
                    if len(sweep_range) <= self.sweeps_num:
                        sweep_range = list(range(0, len(results['sweeps'])))
                bin_size = len(sweep_range) / (self.sweeps_num + 1)
                choices = [
                    sweep_range[0] + int(np.floor((i + 1) * bin_size))
                    for i in range(self.sweeps_num)
                ]
            else:
                if np.random.random() < self.prob:
                    if self.sweep_range[1] <= len(results['sweeps']):
                        sweep_range = list(
                            range(self.sweep_range[0], self.sweep_range[1]))
                    elif self.sweep_range[0] >= len(results['sweeps']):
                        sweep_range = list(range(0, len(results['sweeps'])))
                    else:
                        sweep_range = list(
                            range(self.sweep_range[0], len(results['sweeps'])))
                        if len(sweep_range) <= self.sweeps_num:
                            sweep_range = list(
                                range(0, len(results['sweeps'])))
                    choices = np.random.choice(
                        sweep_range, self.sweeps_num, replace=False)
                else:
                    bin_size = len(results['sweeps']) / (self.sweeps_num + 1)
                    choices = [
                        int(np.floor((i + 1) * bin_size))
                        for i in range(self.sweeps_num)
                    ]
            choices = sorted(choices)
            for idx in choices:
                sweep_idx = min(idx, len(results['sweeps']) - 1)
                sweep = results['sweeps'][sweep_idx]
                if 'lidar_path' in sweep:
                    results['pts_filename'] += [sweep['lidar_path']]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = results['sweeps'][sweep_idx - 1]
                results['filename'].extend(
                    [sweep[sensor]['data_path'] for sensor in self.sensors])
                tmp = [
                    mmcv.imread(sweep[sensor]['data_path'], self.color_type)
                    for sensor in self.sensors
                ]
                img = np.stack(tmp, axis=-1)
                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [
                    lidar_timestamp - sweep[sensor]['timestamp'] / 1e6
                    for sensor in self.sensors
                ]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    results['lidar2img'].append(sweep[sensor]['lidar2img'])
                    results['intrinsics'].append(sweep[sensor]['intrinsics'])
                    results['extrinsics'].append(sweep[sensor]['extrinsics'])
        results['img'] = sweep_imgs_list
        results['timestamp'] = timestamp_imgs_list

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str
