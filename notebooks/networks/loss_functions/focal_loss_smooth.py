import torch
from torch.nn import Module
from torch.nn import functional as F


class FocalLossSmooth(Module):
    """
    Label smoothing added to FocalLoss: https://arxiv.org/abs/1708.02002.
    Basically, we tell the model to not be too confident about its predictions.
    """

    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.1):
        super(FocalLossSmooth, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        with (
            torch.no_grad()
        ):  # For efficiency reasons; since the targets are fixed we don't need to build computational graph for it during learning
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # prevents nans when probability is 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
