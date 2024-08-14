import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yaml

from dataset import get_celeba_dataloader
from network import MobileNet

import torch
from loss_orthogonal import OrthoLoss
from loss_orthogonal import generate_celeba_sensitive_subspace

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
    model_a_logger_path = "./runs_celeba/_paper_fcro_model_a/version_0"
    model_a_path = "model/best_model.pt"
    model_a_path = os.path.join(model_a_logger_path, model_a_path)
    model_a = torch.load(model_a_path, map_location="cpu")["model"]
    ####################################################
    num_group = 2 ** len(config["datamodule"]["sensitive_attributes"])
    device = config["trainer"]["device"]
    ####################################################
    # loss function
    U_list = generate_celeba_sensitive_subspace(
        dataloader=train_dataloader,
        model_a=model_a,
        device=device,
        conditional=config["loss"]["col_loss"]["conditional"],
        moving_base=config["loss"]["col_loss"]["moving_base"],
        threshold=config["loss"]["col_loss"]["threshold"],
        tau=config["metric"]["tau"],
    )
    base_loss_fn = torch.nn.BCELoss()
    ortho_loss_fn = OrthoLoss(
        config=config,
        base_loss_function=base_loss_fn,
        U_list=U_list,
    )
    ####################################################
    epochs = config["trainer"]["epochs"]
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
    model_a.to(device)
    prev_acc = 0
    for epoch in range(epochs):
        ##################################################################################
        # training
        model.train()
        model_a.eval()
        for image, target, _ in tqdm(train_dataloader):
            image = image.to(device)
            target = target.type(torch.float32).to(device)

            pred, feature = model(image)
            with torch.no_grad():
                _, feature_a = model_a(image)

            loss_dict = ortho_loss_fn(
                pred,
                feature,
                feature_a,
                target,
                epoch,
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
    # python ./src/task/celeba/train_fcro_model_t.py
    config_path = "./src/task/celeba/config/config_fcro_model_t.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    main(config)
