import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yaml
from dataset import get_celeba_dataloader

from network import MobileNet

import torch

from utils import set_loggers_folder
from utils import log_tensorboard

from torchmetrics.classification import MulticlassAccuracy

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
    num_group = 2 ** len(config["datamodule"]["sensitive_attributes"])
    device = config["trainer"]["device"]
    ####################################################
    # loss function
    base_loss_weight = config["loss"]["base_loss"]["weight"]
    base_loss_fn = torch.nn.CrossEntropyLoss()
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
    performance_metrics.append(MulticlassAccuracy(num_classes=num_class).to(device))
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
        for image, _, group in tqdm(train_dataloader):
            image = image.to(device)
            group = group.type(torch.int64).to(device)

            pred, _ = model(image)

            base_loss = base_loss_fn(pred, group.squeeze())
            total_loss = base_loss_weight * base_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        ##################################################################################
        # validation
        model.eval()
        with torch.no_grad():
            for image, _, group in tqdm(valid_dataloader):
                image = image.to(device)
                group = group.type(torch.int8).to(device)

                pred, _ = model(image)

                for metric in performance_metrics:
                    metric.update(pred, group.squeeze())
            ##################################################################################
            # valid metrics
            accuracy = log_tensorboard(
                writer,
                performance_metrics,
                None,
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
    # python ./src/task/celeba/train_fcro_model_a.py
    config_path = "./src/task/celeba/config/config_fcro_model_a.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    main(config)
