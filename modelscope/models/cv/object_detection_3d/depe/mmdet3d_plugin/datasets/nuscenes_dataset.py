"""
The implementation here is modified based on PETR, originally Apache-2.0 license and publicly avaialbe at
https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/datasets
"""
import numpy as np
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.
    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, idx_range=None, **kwargs):
        self.idx_range = idx_range
        super().__init__(**kwargs)
        if idx_range is not None:
            assert isinstance(idx_range, (tuple, list))
            assert len(idx_range) == 2
            assert idx_range[0] < idx_range[1]
            assert idx_range[1] <= len(
                self.data_infos
            ), f'the idx_range {idx_range} exceeds total number of dataset:{len(self.data_infos)}'
            self.data_infos = self.data_infos[idx_range[0]:idx_range[1]]

    def get_data_info(self, index):
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=[info['lidar_path']],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            extrinsics_sweep = []
            img_timestamp = []
            for cam_type, cam_info in info['cams'].items():
                img_timestamp.append(cam_info['timestamp'] / 1e6)
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
                extrinsics_sweep.append(None)  # placeholder for sweeps

            input_dict.update(
                dict(
                    img_timestamp=img_timestamp,
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    extrinsics_sweep=extrinsics_sweep,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        return input_dict
