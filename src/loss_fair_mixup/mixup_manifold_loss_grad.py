import torch
import torch.nn as nn
import numpy as np


class MixupManifoldLossGrad(nn.Module):
    def __init__(self):
        super().__init__()

    def _cal_manifold_loss_grad(
        self, model: nn.Module, batch_image: torch.Tensor, batch_group: torch.Tensor
    ):
        num_group = int(batch_group.max().item() + 1)

        lam = np.random.dirichlet(np.ones(num_group), size=1).tolist()[0]
        group_image_list = []
        group_len_list = []
        for i in range(num_group):
            group_image = batch_image[torch.where(batch_group == i)[0]]
            if group_image.shape[0] != 0:
                group_image_list.append(group_image)
                group_len_list.append(group_image.shape[0])

        min_len = min(group_len_list)

        for i in range(len(group_len_list)):
            group_image_list[i] = group_image_list[i][:min_len]

        group_feature_list = []
        for i in range(len(group_len_list)):
            group_feature = model._get_feature(group_image_list[i])
            group_feature_list.append(group_feature)

        inputs_mix = group_feature_list[0] * lam[0]
        for i in range(1, len(group_len_list)):
            inputs_mix += group_feature_list[i] * lam[1]
        inputs_mix = inputs_mix.requires_grad_(True)
        ops = model._get_prediction(inputs_mix).sum()

        gradx = torch.autograd.grad(ops, inputs_mix, create_graph=False)[0].view(
            inputs_mix.shape[0], -1
        )
        loss_grad_list = []
        for i in range(0, len(group_len_list) - 1):
            for j in range(i + 1, len(group_len_list)):
                inputs_1 = group_feature_list[i]
                inputs_0 = group_feature_list[j]

                x_d = (inputs_1 - inputs_0).view(inputs_mix.shape[0], -1)
                grad_inn = (gradx * x_d).sum(1).view(-1)
                loss_grad = torch.abs(grad_inn.mean())
                loss_grad_list.append(loss_grad)

        loss_grad = torch.stack(loss_grad_list).mean()
        return loss_grad

    def forward(self, model, batch_image, batch_group):
        return self._cal_manifold_loss_grad(model, batch_image, batch_group)
