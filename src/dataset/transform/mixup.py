import torch
import numpy as np


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# class Mixup(nn.Module):
#     def __init__(self, num_group: int, mode: str):
#         super().__init__()
#         self._num_group = num_group
#         self._mode = mode

#     def forward(self, batch_image, batch_target, batch_group):
#         group_image_list = []
#         group_target_list = []
#         group_len_list = []
#         for i in range(self._num_group):
#             group_image = batch_image[torch.where(batch_group == i)[0]]
#             group_target = batch_target[torch.where(batch_group == i)[0]]

#             group_image_list.append(group_image)
#             group_len_list.append(group_image.shape[0])
#             group_target_list.append(group_target)

#         lambdas = torch.rand(4)
#         lambdas = lambdas / torch.sum(lambdas)
#         min_len = len(group_len_list)
#         mixed_image = lambdas[i] * group_image_list[i][:min_len]
#         for i in range(1, self._num_group):
#             mixed_image += lambdas[i] * group_image_list[i][:min_len]

#         return self._cal_wasserstein_distance(batch_pred, batch_group)
