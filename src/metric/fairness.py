import torch
import torch.nn as nn
from torchmetrics import ConfusionMatrix


class DP(nn.Module):
    """statistical demographic disparity"""

    def __init__(
        self,
        num_class: int,
        num_group: int,
        tau: float = 0.5,
    ) -> None:
        super().__init__()
        self._num_class = num_class
        self._num_group = num_group
        self._tau = tau
        self._confusionmatrix = ConfusionMatrix(
            task="binary",
            threshold=self._tau,
            num_classes=self._num_class,
        )

    def forward(self, pred, target, group):
        parities = []
        for group_idx in range(self._num_group):
            group_pred = pred[torch.where(group == group_idx)]
            group_target = target[torch.where(group == group_idx)]

            conf_matrix = self._confusionmatrix(group_pred, group_target)
            # RPP : Rate of Positive Predictions = FP + TP / FN + FP + TN + TP
            FP_TP = conf_matrix[0, 1] + conf_matrix[1, 1]
            TN_FP_FN_TP = (
                conf_matrix[0, 0]
                + conf_matrix[0, 1]
                + conf_matrix[1, 0]
                + conf_matrix[1, 1]
            )
            group_parity = FP_TP / (TN_FP_FN_TP + 1e-10)
            parities.append(group_parity)

        disparities = []
        for i in range(self._num_group - 1):
            for j in range(i + 1, self._num_group):
                disparity = torch.abs(parities[i] - parities[j])
                disparities.append(disparity)
        disparities = torch.stack(disparities)

        output_mean = disparities.mean()
        output_max = disparities.max()

        return output_mean, output_max


class SPDD(nn.Module):
    def __init__(
        self,
        num_class: int,
        num_group: int,
        interval: float = 0.02,
    ) -> None:
        super().__init__()
        self._num_class = num_class
        self._num_group = num_group
        self._tau = torch.arange(0, 1, interval)
        self._confusionmatrix = ConfusionMatrix(
            task="binary",
            num_classes=self._num_class,
        )

    def forward(self, pred, target, group):
        parities = []
        for group_idx in range(self._num_group):
            group_pred = pred[torch.where(group == group_idx)]
            group_target = target[torch.where(group == group_idx)]

            group_parity_tau = []
            for tau in self._tau:
                group_pred_binary = torch.where(group_pred > tau, 1, 0)
                conf_matrix = self._confusionmatrix(group_pred_binary, group_target)

                FP_TP = conf_matrix[0, 1] + conf_matrix[1, 1]
                TN_FP_FN_TP = (
                    conf_matrix[0, 0]
                    + conf_matrix[0, 1]
                    + conf_matrix[1, 0]
                    + conf_matrix[1, 1]
                )
                group_parity = FP_TP / (TN_FP_FN_TP + 1e-10)
                group_parity_tau.append(group_parity)
            group_parity_tau = torch.stack(group_parity_tau).mean()
            parities.append(group_parity_tau)

        disparities = []
        for i in range(self._num_group - 1):
            for j in range(i + 1, self._num_group):
                disparity = torch.abs(parities[i] - parities[j])
                disparities.append(disparity)
        disparities = torch.stack(disparities)

        output_mean = disparities.mean()
        output_max = disparities.max()

        return output_mean, output_max


class GS(nn.Module):
    def __init__(
        self,
        num_class: int,
        num_group: int,
        tau: float,
    ) -> None:
        super().__init__()
        self._num_class = num_class
        self._num_group = num_group
        self._tau = tau
        self._confusionmatrix = ConfusionMatrix(
            task="binary",
            threshold=self._tau,
            num_classes=self._num_class,
        )

    def forward(self, pred, target, group):
        parities = []

        # E[Y|f(X)]
        conf_matrix = self._confusionmatrix(pred, target)
        pred_1 = conf_matrix[0, 1] + conf_matrix[1, 1]
        true_1 = conf_matrix[1, 0] + conf_matrix[1, 1]
        gs = true_1 / (pred_1 + 1e-10)
        parities.append(gs)

        # E[Y|f(X),A]
        for group_idx in range(self._num_group):
            group_pred = pred[torch.where(group == group_idx)]
            group_target = target[torch.where(group == group_idx)]

            conf_matrix = self._confusionmatrix(group_pred, group_target)

            pred_1 = conf_matrix[0, 1] + conf_matrix[1, 1]
            true_1 = conf_matrix[1, 0] + conf_matrix[1, 1]
            gs = true_1 / (pred_1 + 1e-10)
            parities.append(gs)
        parities = torch.stack(parities)

        disparities = []
        for i in range(1, self._num_group):
            disparity = torch.abs(parities[0] - parities[i])
            disparities.append(disparity)
        disparities = torch.stack(disparities)

        output_mean = disparities.mean()
        output_max = disparities.max()

        return output_mean, output_max


# only consider TPR
class Equal_Opportunity(nn.Module):
    def __init__(
        self,
        num_class: int,
        num_group: int,
        tau: float = 0.5,
    ) -> None:
        super().__init__()
        self._num_class = num_class
        self._num_group = num_group
        self._tau = tau
        self._confusionmatrix = ConfusionMatrix(
            task="binary",
            threshold=self._tau,
            num_classes=self._num_class,
        )

    def forward(self, pred, target, group):
        parities = []
        for group_idx in range(self._num_group):
            group_pred = pred[torch.where(group == group_idx)]
            group_target = target[torch.where(group == group_idx)]

            conf_matrix = self._confusionmatrix(group_pred, group_target)
            # TPR : True Positive Rate = TP / TP + FN
            TP = conf_matrix[1, 1]
            TP_FN = conf_matrix[1, 1] + conf_matrix[1, 0]
            TPR = TP / (TP_FN + 1e-10)
            parities.append(TPR)

        disparities = []
        for i in range(self._num_group - 1):
            for j in range(i + 1, self._num_group):
                disparity = torch.abs(parities[i] - parities[j])
                disparities.append(disparity)
        disparities = torch.stack(disparities)

        output_mean = disparities.mean()
        output_max = disparities.max()

        return output_mean, output_max


# consider TPR and TNR
class Equalized_Odds(nn.Module):
    def __init__(
        self,
        num_class: int,
        num_group: int,
        tau: float = 0.5,
    ) -> None:
        super().__init__()
        self._num_class = num_class
        self._num_group = num_group
        self._tau = tau
        self._confusionmatrix = ConfusionMatrix(
            task="binary",
            threshold=self._tau,
            num_classes=self._num_class,
        )

    def forward(self, pred, target, group):
        parities_TPR = []
        parities_TNR = []
        for group_idx in range(self._num_group):
            group_pred = pred[torch.where(group == group_idx)]
            group_target = target[torch.where(group == group_idx)]

            conf_matrix = self._confusionmatrix(group_pred, group_target)
            # TPR : True Positive Rate = TP / TP + FN
            TP = conf_matrix[1, 1]
            TP_FN = conf_matrix[1, 1] + conf_matrix[1, 0]
            TPR = TP / (TP_FN + 1e-10)
            parities_TPR.append(TPR)

            # TNR : True Negative Rate = TN / TN + FP
            TN = conf_matrix[0, 0]
            TN_FP = conf_matrix[0, 0] + conf_matrix[0, 1]
            TNR = TN / (TN_FP + 1e-10)
            parities_TNR.append(TNR)

        disparities = []
        for i in range(self._num_group - 1):
            for j in range(i + 1, self._num_group):
                disparity_TPR = torch.abs(parities_TPR[i] - parities_TPR[j])
                disparity_TNR = torch.abs(parities_TNR[i] - parities_TNR[j])
                disparity = (disparity_TPR + disparity_TNR) / (2 + 1e-10)
                disparities.append(disparity)
        disparities = torch.stack(disparities)

        output_mean = disparities.mean()
        output_max = disparities.max()

        return output_mean, output_max
