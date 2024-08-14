import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


def generate_celeba_sensitive_subspace(
    dataloader: DataLoader,
    model_a: nn.Module,
    device: int,
    conditional: bool,
    moving_base: bool,
    threshold: float = 0.99,
    tau: float = 0.5,
):
    print("generate celeba sensitive subspace!")
    print("col conditional :", conditional)
    print("col moving base :", moving_base)
    U_list = []
    if moving_base == True:
        if conditional == True:
            U_list = [None, None]
        else:
            U_list = [None]
    elif moving_base == False:
        emb = []
        targets = []
        model_a.to(device)
        model_a.eval()
        with torch.no_grad():
            for image, target, _ in tqdm(dataloader):
                image = image.to(device)
                feature = model_a._get_feature(image)
                emb.append(feature)
                targets.append(target)

        emb = torch.concat(emb, dim=0).cpu()
        targets = torch.concat(targets, dim=0).squeeze().cpu()

        for i in range(int(conditional) + 1):
            if conditional:
                if i == 0:
                    indices = torch.where(targets < tau)[0]
                elif i == 1:
                    indices = torch.where(targets >= tau)[0]
                emb_sub = torch.index_select(emb, 0, indices)
            else:
                emb_sub = emb

            U, S, _ = torch.linalg.svd(emb_sub.T, full_matrices=False)

            sval_ratio = (S**2) / (S**2).sum()
            r = (torch.cumsum(sval_ratio, -1) <= threshold).sum()
            U_list.append(U[:, :r])

    try:
        for i, u in enumerate(U_list):
            print(i, u.shape)
    except:
        for i, u in enumerate(U_list):
            print(i, u)

    return U_list
