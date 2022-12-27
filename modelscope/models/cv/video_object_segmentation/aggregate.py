# Adopted from https://github.com/Limingxing00/RDE-VOS-CVPR2022
# under MIT License

import torch
import torch.nn.functional as F


# Soft aggregation from STM
def aggregate(prob, keep_bg=False):
    # Caclulate the probability of the background.
    background_prob = torch.prod(1 - prob, dim=0, keepdim=True)
    # Concatenate the probabilities of background and foreground objects.
    new_prob = torch.cat([background_prob, prob], 0).clamp(1e-7, 1 - 1e-7)

    # logit function
    logits = torch.log((new_prob / (1 - new_prob)))

    if keep_bg:
        return F.softmax(logits, dim=0)
    else:
        return F.softmax(logits, dim=0)[1:]


if __name__ == '__main__':
    prob = torch.randn(size=(1, 2, 1, 1))
    prob = torch.sigmoid(prob)
    new = aggregate(prob, keep_bg=True)
    print(prob)
    print(new)
