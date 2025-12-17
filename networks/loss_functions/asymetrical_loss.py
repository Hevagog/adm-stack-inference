import torch
from torch.nn import Module


class AsymmetricLoss(Module):
    """https://arxiv.org/abs/2009.14119"""

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """
        x: input logits
        y: targets (multi-label binarized vector)
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping (for negatives)
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross Entropy Calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Combined Basic Loss
        loss = -1 * (los_pos + los_neg)

        # Asymmetric Focusing (The Exponent)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)

            # Calculate p_t (probability of the ground truth class)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1

            # Select the gamma corresponding to the target (pos or neg)
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)

            # Apply the focusing factor: (1 - pt)^gamma
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)

            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)

            # Multiply the basic loss by the focusing weight
            loss *= one_sided_w

        return loss.sum() / x.size(0)
