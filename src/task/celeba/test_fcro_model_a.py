import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yaml

from dataset import get_celeba_dataloader

import torch

from torchmetrics.classification import MulticlassAccuracy

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
    performance_metrics.append(MulticlassAccuracy(num_classes=num_class).to(device))
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
        for image, _, group in tqdm(test_dataloader):
            image = image.to(device)
            group = group.type(torch.int8).to(device)

            pred, _ = model(image)

            for metric in performance_metrics:
                metric.update(pred, group.squeeze())
    ##################################################################################
    results = {}
    results["epoch"] = epoch
    for metric in performance_metrics:
        key = metric._get_name()
        value = metric.compute()
        results[key] = round(value.item(), 3)

    print(results)

    logger_path = config["logger_path"]
    filepath = os.path.join("./", logger_path, "test_results_alter.json")
    with open(filepath, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # conda activate fairness
    # python ./src/task/celeba/test_fcro_model_a.py
    logger_path = "./runs_celeba/_paper_fcro_model_a/version_1"
    config_path = os.path.join(logger_path, "config.yaml")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    config["logger_path"] = logger_path
    print(logger_path)
    model_path = os.path.join(logger_path, "model/best_model.pt")
    config["model_path"] = model_path
    print(model_path)
    main(config)
