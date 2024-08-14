import numpy as np
import torch
import torch.nn as nn


class RowOrthLoss(nn.Module):
    def __init__(
        self,
        margin: float = 0.0,
    ):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        t_list: list[torch.Tensor],
        a_list: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0.0
        for i, (feature_t_sub, feature_a_sub) in enumerate(zip(t_list, a_list)):
            feature_t_sub = feature_t_sub - feature_t_sub.mean(0, keepdim=True)
            feature_a_sub = feature_a_sub - feature_a_sub.mean(0, keepdim=True)

            sigma = torch.matmul(feature_t_sub.T, feature_a_sub)

            sigma_loss = torch.clamp(torch.sum(sigma**2) - self.margin, min=0)
            loss = loss + sigma_loss / sigma.numel()

        loss = loss / len(t_list)

        return loss
