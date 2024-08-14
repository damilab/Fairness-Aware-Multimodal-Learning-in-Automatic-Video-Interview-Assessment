import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yaml

from dataset import get_celeba_dataloader
from network import MobileNet
from network.classifier import MobileNetV3SmallDomainClassifier

import torch
import numpy as np
from utils import set_loggers_folder
from utils import log_tensorboard

from torchmetrics.classification import BinaryAccuracy
from metric import DemographicParity

from tqdm import tqdm

from itertools import chain
import torch.nn as nn
from typing import Iterable


def concat_net_params(*net: nn.Module) -> Iterable[nn.Parameter]:
    return chain(*map(lambda x: x.parameters(), net))


def main(config: dict):
    ####################################################
    # dataset & dataloader
    train_dataloader, valid_dataloader, _ = get_celeba_dataloader(config)
    ####################################################
    # model
    model = MobileNet(
        num_class=config["module"]["model"]["num_class"],
        type=config["module"]["model"]["type"],
    )
    domain_classifier = MobileNetV3SmallDomainClassifier(
        num_class=config["module"]["classifier"]["num_class"],
    )
    ####################################################
    # batch-pre-processing
    num_group = 2 ** len(config["datamodule"]["sensitive_attributes"])
    device = config["trainer"]["device"]
    ####################################################
    # loss function
    base_loss_weight = config["loss"]["base_loss"]["weight"]
    base_loss_fn = torch.nn.BCELoss()
    domain_loss_fn = torch.nn.CrossEntropyLoss()
    dann_gamma = config["loss"]["domain_loss"]["dann_gamma"]
    ####################################################
    epochs = config["trainer"]["epochs"]
    ####################################################
    # optim
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["trainer"]["learning_rate"],
    )
    classifier_param = concat_net_params(model._encoder, domain_classifier)
    optimizer_domain = torch.optim.Adam(
        classifier_param,
        lr=config["trainer"]["learning_rate_domain"],
    )
    ####################################################
    # metrics
    num_class = config["metric"]["num_class"]
    tau = config["metric"]["tau"]
    interval = config["metric"]["interval"]
    performance_metrics = []
    performance_metrics.append(BinaryAccuracy().to(device))

    fairness_metrics = []
    fairness_metrics.append(DemographicParity(tau, num_group).to(device))
    ####################################################
    # loggers
    set_loggers = set_loggers_folder(
        config=config,
        logger_root=config["trainer"]["logger"]["root"],
        logger_name=config["trainer"]["logger"]["name"],
    )
    logger_path = set_loggers["logger_path"]
    model_path = set_loggers["model_path"]
    writer = set_loggers["writer"]
    ####################################################
    model.to(device)
    domain_classifier.to(device)
    prev_acc = 0
    for epoch in range(epochs):
        ##################################################################################
        # training
        model.train()
        domain_classifier.train()
        p = epoch / epochs
        lambd = (2.0 / (1.0 + np.exp(-dann_gamma * p))) - 1
        for image, target, group in tqdm(train_dataloader):
            image = image.to(device)
            target = target.type(torch.float32).to(device)
            group = group.type(torch.int64).to(device)

            pred, _ = model(image)

            base_loss = base_loss_fn(pred, target)
            total_loss = base_loss_weight * base_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            domain_classifier.update_lambd(lambd)

            feature = model._get_feature(image)
            group_pred = domain_classifier(feature)

            domain_loss = domain_loss_fn(group_pred, group.squeeze())

            optimizer_domain.zero_grad()
            domain_loss.backward()
            optimizer_domain.step()
        ##################################################################################
        # validation
        model.eval()
        domain_classifier.eval()
        with torch.no_grad():
            for image, target, group in tqdm(valid_dataloader):
                image = image.to(device)
                target = target.type(torch.int8).to(device)
                group = group.type(torch.int8).to(device)

                pred, _ = model(image)

                for metric in performance_metrics:
                    metric.update(pred, target)

                for metric in fairness_metrics:
                    metric.update(pred, target, group)
            ##################################################################################
            # valid metrics
            accuracy = log_tensorboard(
                writer,
                performance_metrics,
                fairness_metrics,
                epoch,
                s="valid",
            )

            if prev_acc < accuracy:
                prev_acc = accuracy
                filepath = os.path.join(model_path, "best_model.pt")
                print("filepath :", filepath)
                save_dict = {
                    "epoch": epoch,
                    "model": model,
                }
                torch.save(save_dict, filepath)
        ##################################################################################

    writer.flush()
    writer.close()


if __name__ == "__main__":
    # conda activate fairness
    # python ./src/task/celeba/train_adv.py
    config_path = "./src/task/celeba/config/config_adv.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    config["loss"]["domain_loss"]["dann_gamma"] = 1
    main(config)

    config["loss"]["domain_loss"]["dann_gamma"] = 2
    main(config)

    config["loss"]["domain_loss"]["dann_gamma"] = 3
    main(config)

    config["loss"]["domain_loss"]["dann_gamma"] = 4
    main(config)

    config["loss"]["domain_loss"]["dann_gamma"] = 5
    main(config)

    config["loss"]["domain_loss"]["dann_gamma"] = 6
    main(config)

    config["loss"]["domain_loss"]["dann_gamma"] = 7
    main(config)

    config["loss"]["domain_loss"]["dann_gamma"] = 8
    main(config)

    config["loss"]["domain_loss"]["dann_gamma"] = 9
    main(config)

    config["loss"]["domain_loss"]["dann_gamma"] = 10
    main(config)
