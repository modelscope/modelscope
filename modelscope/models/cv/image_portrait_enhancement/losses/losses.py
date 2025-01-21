# The GPEN implementation is also open-sourced by the authors,
# and available at https://github.com/yangxy/GPEN/tree/main/training/loss/id_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_irse import Backbone


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}'
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * F.l1_loss(
            pred, target, reduction=self.reduction)


class IDLoss(nn.Module):

    def __init__(self, model_path, device='cuda', ckpt_dict=None):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6,
            mode='ir_se').to(device)
        if ckpt_dict is None:
            self.facenet.load_state_dict(
                torch.load(model_path, map_location=torch.device('cpu')))
        else:
            self.facenet.load_state_dict(ckpt_dict)
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        _, _, h, w = x.shape
        assert h == w
        if h != 256:
            x = self.pool(x)
        x = x[:, :, 35:-33, 32:-36]  # crop roi
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    @torch.no_grad()
    def forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({
                'diff_target': float(diff_target),
                'diff_input': float(diff_input),
                'diff_views': float(diff_views)
            })
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs
