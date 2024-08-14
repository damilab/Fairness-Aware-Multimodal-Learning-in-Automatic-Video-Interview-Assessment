import os
import yaml

from dataset import get_celeba_fscl_dataloader
from network import MobileNet

import torch
import torch.nn.functional as F
from loss_fscl import FairSupConLoss

from utils import set_loggers_folder
from utils import log_tensorboard

from torchmetrics.classification import BinaryAccuracy
from metric import DemographicParity

from tqdm import tqdm


def main(config: dict):
    ####################################################
    # dataset & dataloader
    train_fscl_dataloader, train_dataloader, valid_dataloader, _ = (
        get_celeba_fscl_dataloader(config)
    )
    ####################################################
    # model
    model = MobileNet(
        num_class=config["module"]["model"]["num_class"],
        type=config["module"]["model"]["type"],
    )
    ####################################################
    # batch-pre-processing
    num_group = 2 ** len(config["datamodule"]["sensitive_attributes"])
    device = config["trainer"]["device"]
    ####################################################
    # loss function
    fscl_loss_fn = FairSupConLoss(
        temperature=config["loss"]["fscl_loss"]["temperature"],
        base_temperature=config["loss"]["fscl_loss"]["base_temperature"],
        group_norm=config["loss"]["fscl_loss"]["group_norm"],
        method=config["loss"]["fscl_loss"]["method"],
    )
    base_loss_weight = config["loss"]["base_loss"]["weight"]
    base_loss_fn = torch.nn.BCELoss()
    ####################################################
    epochs = config["trainer"]["epochs"]
    ####################################################
    # optim
    optimizer_encoder = torch.optim.Adam(
        model._encoder.parameters(),
        lr=config["trainer"]["learning_rate"],
    )
    optimizer_classifier = torch.optim.Adam(
        model._classifier.parameters(),
        lr=config["trainer"]["learning_rate"],
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
    prev_acc = 0
    for epoch in range(epochs):
        ##################################################################################
        # training
        model.train()
        print("encoder training")
        for image, target, group in tqdm(train_fscl_dataloader):
            image = torch.cat([image[0], image[1]], dim=0)
            image = image.to(device)
            target = target.type(torch.int64).squeeze().to(device)
            group = group.type(torch.int64).squeeze().to(device)

            bsz = target.shape[0]

            pred, feature = model(image)

            feature = model._get_feature(image)
            feature = F.normalize(feature, dim=1)
            f1, f2 = torch.split(feature, [bsz, bsz], dim=0)
            feature = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            fscl_loss = fscl_loss_fn(
                feature,
                target,
                group,
            )

            optimizer_encoder.zero_grad()
            fscl_loss.backward()
            optimizer_encoder.step()

        print("classifier training")
        for image, target, group in tqdm(train_dataloader):
            image = image.to(device)
            target = target.type(torch.float32).to(device)

            pred, _ = model(image)
            base_loss = base_loss_fn(pred, target)
            base_loss = base_loss_weight * base_loss

            optimizer_classifier.zero_grad()
            base_loss.backward()
            optimizer_classifier.step()
        ##################################################################################
        # validation
        model.eval()
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


if __name__ == "__main__":
    # conda activate fairness
    # python ./src/task/celeba/train_fscl_contrastive.py
    config_path = "./src/task/celeba/config/config_fscl_contrastive.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    main(config)
