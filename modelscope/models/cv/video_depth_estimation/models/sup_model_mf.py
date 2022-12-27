# Part of the implementation is borrowed and modified from PackNet-SfM,
# made publicly available under the MIT License at https://github.com/TRI-ML/packnet-sfm
from modelscope.models.cv.video_depth_estimation.models.model_utils import \
    merge_outputs
from modelscope.models.cv.video_depth_estimation.models.sfm_model_mf import \
    SfmModelMF
from modelscope.models.cv.video_depth_estimation.utils.depth import depth2inv


class SupModelMF(SfmModelMF):
    """
    Model that inherits a depth and pose network from SfmModel and
    includes the photometric loss for self-supervised training.

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """

    def __init__(self, **kwargs):
        # Initializes SfmModel
        super().__init__(**kwargs)
        # Initializes the photometric loss

        self._network_requirements = {
            'depth_net': True,  # Depth network required
            'pose_net': False,  # Pose network required
            'percep_net': False,  # Pose network required
        }

        self._train_requirements = {
            'gt_depth': True,  # No ground-truth depth required
            'gt_pose': True,  # No ground-truth pose required
        }

        # self._photometric_loss = MultiViewPhotometricLoss(**kwargs)
        # self._loss = SupervisedDepthPoseLoss(**kwargs)

    @property
    def logs(self):
        """Return logs."""
        return {**super().logs, **self._photometric_loss.logs}

    def supervised_loss(self,
                        image,
                        ref_images,
                        inv_depths,
                        gt_depth,
                        gt_poses,
                        poses,
                        intrinsics,
                        return_logs=False,
                        progress=0.0):
        """
        Calculates the self-supervised photometric loss.

        Parameters
        ----------
        image : torch.Tensor [B,3,H,W]
            Original image
        ref_images : list of torch.Tensor [B,3,H,W]
            Reference images from context
        inv_depths : torch.Tensor [B,1,H,W]
            Predicted inverse depth maps from the original image
        poses : list of Pose
            List containing predicted poses between original and context images
        intrinsics : torch.Tensor [B,3,3]
            Camera intrinsics
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar a "metrics" dictionary
        """
        return self._loss(
            image,
            ref_images,
            inv_depths,
            depth2inv(gt_depth),
            gt_poses,
            intrinsics,
            intrinsics,
            poses,
            return_logs=return_logs,
            progress=progress)

    def forward(self, batch, return_logs=False, progress=0.0):
        """
        Processes a batch.

        Parameters
        ----------
        batch : dict
            Input batch
        return_logs : bool
            True if logs are stored
        progress :
            Training progress percentage

        Returns
        -------
        output : dict
            Dictionary containing a "loss" scalar and different metrics and predictions
            for logging and downstream usage.
        """
        # Calculate predicted depth and pose output
        output = super().forward(batch, return_logs=return_logs)
        if not self.training:
            # If not training, no need for self-supervised loss
            return output
        else:
            if output['poses'] is None:
                return None
            # Otherwise, calculate self-supervised loss
            self_sup_output = self.supervised_loss(
                batch['rgb_original'],
                batch['rgb_context_original'],
                output['inv_depths'],
                batch['depth'],
                batch['pose_context'],
                output['poses'],
                batch['intrinsics'],
                return_logs=return_logs,
                progress=progress)
            # Return loss and metrics
            return {
                'loss': self_sup_output['loss'],
                **merge_outputs(output, self_sup_output),
            }
