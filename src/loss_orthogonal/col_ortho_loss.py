import torch
import torch.nn as nn


class ColOrthLoss(nn.Module):
    def __init__(
        self,
        U_list: list,
        margin: float = 0.0,
        threshold: float = 0.99,
        moving_base: bool = True,
        moving_epoch: int = 3,
        device: int = 0,
    ):
        super(ColOrthLoss, self).__init__()
        self.U_list = U_list
        self.margin = margin
        self.threshold = threshold
        self.moving_base = moving_base
        self.moving_epoch = moving_epoch
        self.device = device

    def forward(
        self,
        t_list: list[torch.Tensor],
        a_list: list[torch.Tensor],
        epoch: int,
    ) -> torch.Tensor:
        loss = 0.0
        for i in range(len(t_list)):
            feature_t_sub = t_list[i]
            if self.moving_base == True:
                feature_a_sub = a_list[i]
                if epoch <= self.moving_epoch:
                    if self.U_list[i] is None:
                        U, S, _ = torch.linalg.svd(feature_a_sub.T, full_matrices=False)

                        sval_ratio = (S**2) / (S**2).sum()
                        r = (torch.cumsum(sval_ratio, -1) < self.threshold).sum()

                        self.U_list[i] = U[:, :r]
                    else:
                        with torch.no_grad():
                            self.update_space(feature_a_sub, condition=i)

                U_sub = self.U_list[i]
            elif self.moving_base == False:
                assert self.U_list[i] is not None
                U_sub = self.U_list[i]

            proj_fea = torch.matmul(feature_t_sub, U_sub.to(self.device))
            con_loss = torch.clamp(torch.sum(proj_fea**2) - self.margin, min=0)

            loss = loss + con_loss / feature_t_sub.shape[0]

        loss = loss / len(t_list)

        return loss

    def update_space(
        self,
        feature: torch.Tensor,
        condition: int,
    ):
        bases = self.U_list[condition].clone()

        R2 = torch.matmul(feature.T, feature)
        delta = []
        for ki in range(bases.shape[1]):
            base = bases[:, ki : ki + 1]
            delta_i = torch.matmul(torch.matmul(base.T, R2), base).squeeze()
            delta.append(delta_i)

        delta = torch.hstack(delta)

        _, S_, _ = torch.linalg.svd(feature.T, full_matrices=False)
        sval_total = (S_**2).sum()

        # projection_diff = feature - self.projection(feature, bases)
        projection_diff = (
            feature - torch.matmul(torch.matmul(bases, bases.T), feature.T).T
        )
        U, S, V = torch.linalg.svd(projection_diff.T, full_matrices=False)

        stack = torch.hstack((delta, S**2))
        S_new, sel_index = torch.topk(stack, len(stack))

        r = 0
        accumulated_sval = 0

        for i in range(len(stack)):
            if accumulated_sval < self.threshold * sval_total and r < feature.shape[1]:
                accumulated_sval += S_new[i].item()
                r += 1
            else:
                break

        sel_index = sel_index[:r]
        S_new = S_new[:r]

        Ui = torch.hstack([bases, U])
        U_new = torch.index_select(Ui, 1, sel_index)

        # sel_index_from_new = sel_index[sel_index >= len(delta)]
        # sel_index_from_old = sel_index[sel_index < len(delta)]
        # print(f"from old: {len(sel_index_from_old)}, from new: {len(sel_index_from_new)}")

        self.U_list[condition] = U_new.clone()
