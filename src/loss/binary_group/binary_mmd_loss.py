import torch
import torch.nn as nn


class BinaryMMDLoss(nn.Module):
    """Fully Connected Layer 전에 들어가는 representation 공간에서 Major group의 Z와 Minor group의 Z 간 MMD Loss를 추가"""

    """ Z - hidden space """

    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def gaussian_kernel(
        self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None
    ):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [
            torch.exp(-L2_distance / (bandwidth_temp + 1e-8))
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def _split_feature_by_group(
        self,
        batch_feature: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group_feature_0 = batch_feature[torch.where(batch_group == 0)[0]]
        group_feature_1 = batch_feature[torch.where(batch_group == 1)[0]]
        return group_feature_0, group_feature_1

    def _cal_binary_maximum_mean_discrepancy(
        self,
        group_feature_0: torch.Tensor,
        group_feature_1: torch.Tensor,
    ) -> torch.Tensor:
        min_len = min(len(group_feature_0), len(group_feature_1))
        source = group_feature_0[:min_len]
        target = group_feature_1[:min_len]

        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

    def forward(
        self,
        batch_feature: torch.Tensor,
        batch_group: torch.Tensor,
    ) -> torch.Tensor:
        group_feature_0, group_feature_1 = self._split_feature_by_group(
            batch_feature,
            batch_group,
        )
        mmd_loss = self._cal_binary_maximum_mean_discrepancy(
            group_feature_0,
            group_feature_1,
        )
        return mmd_loss
