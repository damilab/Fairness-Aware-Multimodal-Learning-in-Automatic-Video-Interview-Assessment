import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yaml

from dataset import get_celeba_dataloader
from dataset.transform_multifair import MixupRandomly
from dataset.transform_multifair import MixupInTurn
from dataset.transform_multifair import MixupInDistance
from dataset.transform_multifair import MixupViaInterpolations

from network import MobileNet

import torch

from utils import set_loggers_folder
from utils import log_tensorboard

from torchmetrics.classification import BinaryAccuracy
from metric import DemographicParity

from tqdm import tqdm


def main(config: dict):
    ####################################################
    # dataset & dataloader
    train_dataloader, valid_dataloader, _ = get_celeba_dataloader(config)
    model = MobileNet(
        num_class=config["module"]["model"]["num_class"],
        type=config["module"]["model"]["type"],
    )
    ####################################################
    # mixup
    num_group = 2 ** len(config["datamodule"]["sensitive_attributes"])
    mixup_method = config["datamodule"]["mixup_method"]
    if mixup_method == "mixup_randomly":
        mixup_method = MixupRandomly()
    elif mixup_method == "mixup_in_turn":
        mixup_method = MixupInTurn(num_group)
    elif mixup_method == "mixup_in_distance":
        mixup_method = MixupInDistance(num_group)
    elif mixup_method == "mixup_via_interpolations":
        mixup_method = MixupViaInterpolations(num_group)
    print("mixup_method :", mixup_method._get_name())
    ####################################################
    # loss function
    base_loss_weight = config["loss"]["base_loss"]["weight"]
    base_loss_fn = torch.nn.BCELoss()
    ####################################################
    epochs = config["trainer"]["epochs"]
    device = config["trainer"]["device"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["trainer"]["learning_rate"],
    )
    ####################################################
    # metrics
    num_group = 2 ** len(config["datamodule"]["sensitive_attributes"])
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

            image, target = mixup_method(
                image,
                target,
                group,
            )

            pred, _ = model(image)

            base_loss = base_loss_fn(pred, target)
            base_loss = base_loss_weight * base_loss

            optimizer.zero_grad()
            base_loss.backward()
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
            # valid metrics
            acc = log_tensorboard(
                writer,
                performance_metrics,
                fairness_metrics,
                epoch,
                s="valid",
            )

            if prev_acc < acc:
                prev_acc = acc
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
    # python ./src/task/celeba/train_multifair.py
    config_path = "./src/task/celeba/config/config_multifair.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    config["datamodule"]["mixup_method"] = "mixup_randomly"
    main(config)
    config["datamodule"]["mixup_method"] = "mixup_in_turn"
    main(config)
    config["datamodule"]["mixup_method"] = "mixup_in_distance"
    main(config)
    config["datamodule"]["mixup_method"] = "mixup_via_interpolations"
    main(config)