import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yaml

from dataset import get_celeba_dataloader
from dataset.transform_in_turn import ImageInTurn
from dataset.transform_sampling import ImageWeightedOverUnderSampling
from network import MobileNet

import torch
from loss import FairnessLoss

from utils import set_loggers_folder
from utils import log_tensorboard

from torchmetrics.classification import BinaryAccuracy
from metric import DemographicParity

from tqdm import tqdm


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
    ####################################################
    # batch-pre-processing
    num_group = 2 ** len(config["datamodule"]["sensitive_attributes"])
    device = config["trainer"]["device"]
    image_in_turn = ImageInTurn(num_group, device)
    image_sampling_method = ImageWeightedOverUnderSampling(
        batch_size=config["datamodule"]["batch_size"],
        tau=config["datamodule"]["weighted_sampling_tau"],
    )
    ####################################################
    # loss function
    base_loss_fn = torch.nn.BCELoss()
    loss_function = FairnessLoss(config, base_loss_fn)
    ####################################################
    epochs = config["trainer"]["epochs"]
    device = config["trainer"]["device"]
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

            image, target, group, group_others = image_in_turn(
                image,
                target,
                group,
            )

            image, target, group = image_sampling_method(
                image,
                target,
                group,
                group_others,
            )

            pred, feature = model(image)

            loss_dict = loss_function(
                pred,
                feature,
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
    # python ./src/task/celeba/train_ours.py
    config_path = "./src/task/celeba/config/config_ours.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    main(config)

    # search tau
    # config["datamodule"]["weighted_sampling_tau"] = 0.0
    # main(config)
    # config["datamodule"]["weighted_sampling_tau"] = 0.1
    # main(config)
    # config["datamodule"]["weighted_sampling_tau"] = 0.2
    # main(config)
    # config["datamodule"]["weighted_sampling_tau"] = 0.3
    # main(config)
    # config["datamodule"]["weighted_sampling_tau"] = 0.4
    # main(config)
    # config["datamodule"]["weighted_sampling_tau"] = 0.5
    # main(config)
    # config["datamodule"]["weighted_sampling_tau"] = 0.6
    # main(config)
    # config["datamodule"]["weighted_sampling_tau"] = 0.7
    # main(config)
    # config["datamodule"]["weighted_sampling_tau"] = 0.8
    # main(config)
    # config["datamodule"]["weighted_sampling_tau"] = 0.9
    # main(config)
    # config["datamodule"]["weighted_sampling_tau"] = 1.0
    # main(config)

    # search lambda
    # config["loss"]["wd_loss"]["weight"] = 0.1
    # main(config)
    # config["loss"]["wd_loss"]["weight"] = 0.2
    # main(config)
    # config["loss"]["wd_loss"]["weight"] = 0.3
    # main(config)
    # config["loss"]["wd_loss"]["weight"] = 0.4
    # main(config)
    # config["loss"]["wd_loss"]["weight"] = 0.5
    # main(config)
    # config["loss"]["wd_loss"]["weight"] = 0.6
    # main(config)
    # config["loss"]["wd_loss"]["weight"] = 0.7
    # main(config)
    # config["loss"]["wd_loss"]["weight"] = 0.8
    # main(config)
    # config["loss"]["wd_loss"]["weight"] = 0.9
    # main(config)
    # config["loss"]["wd_loss"]["weight"] = 1.0
    # main(config)
