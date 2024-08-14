import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yaml
from dataset import get_celeba_dataloader
from network import MobileNet

import torch
from loss_fair_mixup import FairMixupLoss

from utils import set_loggers_folder
from utils import log_tensorboard

from torchmetrics.classification import BinaryAccuracy
from metric import DemographicParity

from tqdm import tqdm


def main(config: dict):
    train_dataloader, valid_dataloader, _ = get_celeba_dataloader(config)
    model = MobileNet(
        num_class=config["module"]["model"]["num_class"],
        type=config["module"]["model"]["type"],
    )
    ####################################################
    num_group = 2 ** len(config["datamodule"]["sensitive_attributes"])
    device = config["trainer"]["device"]
    ####################################################
    # loss function
    base_loss_fn = torch.nn.BCELoss()
    fairness_loss = FairMixupLoss(config, base_loss_fn)
    ####################################################
    epochs = config["trainer"]["epochs"]
    ####################################################
    # optim
    optimizer = torch.optim.Adam(
        model.parameters(),
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
        for image, target, group in tqdm(train_dataloader):
            image = image.to(device)
            target = target.type(torch.float32).to(device)
            group = group.type(torch.float32).to(device)

            loss_dict = fairness_loss(
                model,
                image,
                target,
                group,
            )

            optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            optimizer.step()
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
            pearson = log_tensorboard(
                writer,
                performance_metrics,
                fairness_metrics,
                epoch,
                s="valid",
            )

            if prev_acc < pearson:
                prev_acc = pearson
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
    # python ./src/task/celeba/train_fair_mixup.py
    config_path = "./src/task/celeba/config/config_fair_mixup.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # gapreg
    config["loss"]["gapreg_loss"]["weight"] = 1.0
    config["loss"]["mixup_loss_grad"]["weight"] = 0.0
    config["loss"]["mixup_manifold_loss_grad"]["weight"] = 0.0
    main(config)

    # mixup loss grad
    config["loss"]["gapreg_loss"]["weight"] = 0.0
    config["loss"]["mixup_loss_grad"]["weight"] = 1.0
    config["loss"]["mixup_manifold_loss_grad"]["weight"] = 0.0
    main(config)

    # mixup manifold loss grad
    config["loss"]["gapreg_loss"]["weight"] = 0.0
    config["loss"]["mixup_loss_grad"]["weight"] = 0.0
    config["loss"]["mixup_manifold_loss_grad"]["weight"] = 1.0
    main(config)
