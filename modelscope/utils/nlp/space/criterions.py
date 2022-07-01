import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def compute_kl_loss(p, q, filter_scores=None):
    p_loss = F.kl_div(
        F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(
        F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum(dim=-1)
    q_loss = q_loss.sum(dim=-1)

    # mask is for filter mechanism
    if filter_scores is not None:
        p_loss = filter_scores * p_loss
        q_loss = filter_scores * q_loss

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


class CatKLLoss(_Loss):
    """
    CatKLLoss
    """

    def __init__(self, reduction='mean'):
        super(CatKLLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction

    def forward(self, log_qy, log_py):
        """
        KL(qy|py) = Eq[qy * log(q(y) / p(y))]

        log_qy: (batch_size, latent_size)
        log_py: (batch_size, latent_size)
        """
        qy = torch.exp(log_qy)
        kl = torch.sum(qy * (log_qy - log_py), dim=1)

        if self.reduction == 'mean':
            kl = kl.mean()
        elif self.reduction == 'sum':
            kl = kl.sum()
        return kl
