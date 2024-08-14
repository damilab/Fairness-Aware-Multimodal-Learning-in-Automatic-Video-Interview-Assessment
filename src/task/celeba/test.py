import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yaml
from dataset import get_celeba_dataloader

import torch

from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision
from metric import DemographicParity
from metric import DemographicParity_Fair_Mixup
from metric import Strong_Pairwise_Demographic_Disparity
from metric import Equal_Opportunity
from metric import Strong_Pairwise_Equal_Opportunity
from metric import Equalized_Odds
from metric import Strong_Pairwise_Equalized_Odds

from tqdm import tqdm
import json


def main(config: dict):
    #############################################################
    # test dataset & dataloader
    _, _, test_dataloader = get_celeba_dataloader(config)
    #############################################################
    # metrics
    device = config["trainer"]["device"]
    num_group = 2 ** len(config["datamodule"]["sensitive_attributes"])
    num_class = config["metric"]["num_class"]
    tau = config["metric"]["tau"]
    interval = config["metric"]["interval"]

    performance_metrics = []
    performance_metrics.append(BinaryAccuracy().to(device))
    performance_metrics.append(BinaryAveragePrecision().to(device))

    fairness_metrics = []
    fairness_metrics.append(DemographicParity(tau, num_group).to(device))
    fairness_metrics.append(DemographicParity_Fair_Mixup(num_group).to(device))
    fairness_metrics.append(
        Strong_Pairwise_Demographic_Disparity(interval, num_group).to(device)
    )

    fairness_metrics.append(Equal_Opportunity(tau, num_group).to(device))
    fairness_metrics.append(
        Strong_Pairwise_Equal_Opportunity(interval, num_group).to(device)
    )

    fairness_metrics.append(Equalized_Odds(tau, num_group).to(device))
    fairness_metrics.append(
        Strong_Pairwise_Equalized_Odds(interval, num_group).to(device)
    )

    #############################################################
    # model
    model_path = config["model_path"]
    save_dict = torch.load(model_path, map_location="cpu")
    epoch = save_dict["epoch"]
    model = save_dict["model"]
    model.to(device)
    model.eval()
    #############################################################
    with torch.no_grad():
        for image, target, group in tqdm(test_dataloader):
            batch_image = image.to(device)
            batch_target = target.type(torch.int8).to(device)
            batch_group = group.type(torch.int8).to(device)

            batch_pred, _ = model(batch_image)

            for metric in performance_metrics:
                metric.update(batch_pred, batch_target)

            for metric in fairness_metrics:
                metric.update(batch_pred, batch_target, batch_group)
    ##################################################################################
    results = {}
    for metric in performance_metrics:
        key = metric._get_name()
        value = metric.compute()
        results[key] = round(value.item(), 4)
    for metric in fairness_metrics:
        key = metric._get_name()
        value_mean, value_max = metric.compute()
        results[key + "_mean"] = round(value_mean.item(), 4)
        results[key + "_max"] = round(value_max.item(), 4)

    print(results)

    logger_path = config["logger_path"]
    filepath = os.path.join("./", logger_path, config["trainer"]["log_file_nema"])
    with open(filepath, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # conda activate fairness
    # python ./src/task/celeba/test.py
    logger_path = "./runs_celeba/_paper_fcro_model_t/version_0"
    config_path = os.path.join(logger_path, "config.yaml")
    print(config_path)
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    config["logger_path"] = logger_path
    print(logger_path)
    model_path = os.path.join(logger_path, "model/best_model.pt")
    config["model_path"] = model_path
    print(model_path)
    config["trainer"]["device"] = 0
    config["trainer"]["log_file_nema"] = "test_results_performance.json"
    main(config)
