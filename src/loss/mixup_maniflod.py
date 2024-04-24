import torch
import torch.nn as nn
import numpy as np
import math
from functools import reduce

class Mixup_manifold(nn.Module):
    def __init__(self, num_group: int, device='cuda:0'):
        super().__init__()
        self._num_group = num_group
        self._device = device
    

    def _cal_loss_grad_mf(self, model: nn.Module, model_linear: nn.Module, batch_image: torch.Tensor , batch_group: torch.Tensor):
        # group별로 split
        group_len_list = []
        group_image_list = []
        for i in range(self._num_group):
            group_image = batch_image[torch.where(batch_group == i)[0]]
            group_len_list.append(len(group_image)) 
            group_image_list.append(group_image)
        
    
        # # prediction oversampling (max_len or lcm)
        # if self._lcm == True:
        #     max_len = reduce(lambda x, y: x * y // math.gcd(x, y), group_len_list)
        # elif self._lcm == False:
        #     max_len = max(group_len_list)


        # expanded_group_image_list = []
        # for idx, group_image in enumerate(group_image_list):
        #     try:  
        #         expansion_factor = (max_len // group_image.shape[0]) + 1
        #         expanded_group_image = group_image.repeat(expansion_factor)[:max_len]
        #         expanded_group_image_list.append(expanded_group_image)
        #     except:
        #         pass

        # calculate loss grad mf
        min_group_len= min(group_len_list)

        lam = np.random.dirichlet(np.ones(self._num_group), size=1)
        lam = torch.from_numpy(lam).float().to(self._device)
        feat_list=[]
        for i in range(self._num_group):
            feat = model(group_image_list[i][:min_group_len])
            feat_list.append(feat)

        
        input_mix = torch.zeros([min_group_len, 512],device=self._device)
        
        for i in range(self._num_group):
            input_mix += lam[0,i]*feat_list[i]
        
        input_mix = input_mix.requires_grad_(True)
      
        ops= model_linear(input_mix).sum()
        
        

        x_d_list=[]
        gradx = torch.autograd.grad(ops, input_mix, create_graph=True)[0].view(input_mix.shape[0], -1)
        for i in range(len(group_image_list) - 1):
            for j in range(i + 1, len(group_image_list)):
                feat_1 = feat_list[i]
                feat_0 = feat_list[j]

                x_d = (feat_1 - feat_0).view(input_mix.shape[0], -1)
                x_d_list.append(x_d)
        grad_inn = (gradx * torch.mean(torch.stack(x_d_list),dim=0)).sum(1).view(-1)
        loss_grad = torch.abs(grad_inn.mean())

        return loss_grad


    def forward(self, model: nn.Module,model_linear: nn.Module ,batch_image: torch.Tensor, batch_group: torch.Tensor):
        return self._cal_loss_grad_mf(model, model_linear, batch_image, batch_group)