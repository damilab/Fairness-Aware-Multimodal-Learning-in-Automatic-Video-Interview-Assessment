import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yaml

from dataset.transform import (
    get_base_train_transform,
    get_base_val_transform,
)
from dataset import CelebADataset_MSA
from torch.utils.data import DataLoader
from network import ResNet, MobileNet_v2

import torch
from loss import FairnessLoss
from loss import L2Loss
from loss import WDLoss
from loss import PredictionOversamplingWDLoss
from loss import GapReg

from utils import set_loggers_folder

from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision
from metric import DP, SPDD, GS
from metric import Equal_Opportunity, Equalized_Odds

from tqdm import tqdm
import json


def main(config: dict):
    ####################################################
    # dataset & dataloader
    train_transform = get_base_train_transform()
    valid_transform = get_base_val_transform()
    train_dataset = CelebADataset_MSA(
        root=config["datamodule"]["image_dir"],
        split="train",
        transform=train_transform,
        gaussian_noise_transform=False,
        download=False,
        target_attribute=config["datamodule"]["target_attribute"],
        sensitive_attributes=config["datamodule"]["sensitive_attributes"],
    )
    valid_dataset = CelebADataset_MSA(
        root=config["datamodule"]["image_dir"],
        split="valid",
        transform=valid_transform,
        gaussian_noise_transform=False,
        download=False,
        target_attribute=config["datamodule"]["target_attribute"],
        sensitive_attributes=config["datamodule"]["sensitive_attributes"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        num_workers=config["datamodule"]["num_workers"],
        batch_size=config["datamodule"]["batch_size"],
        persistent_workers=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        drop_last=False,
        num_workers=config["datamodule"]["num_workers"],
        batch_size=config["datamodule"]["batch_size"],
        persistent_workers=True,
    )
    ####################################################
    # model
    model_type = config["module"]["model"]["type"]
    if model_type == "res18" or model_type == "res34":
        model = ResNet(
            num_class=config["module"]["model"]["num_class"],
            type=config["module"]["model"]["type"],
            pretrained_path=config["module"]["model"]["pretrained_path"],
        )
    elif model_type == "mobilenet":
        model = MobileNet_v2(
            num_class=config["module"]["model"]["num_class"],
        )
    ####################################################
    # loss function
    num_group = 2 ** len(config["datamodule"]["sensitive_attributes"])

    base_loss_weight = config["loss"]["base_loss"]["weight"]
    base_loss_function = torch.nn.BCELoss()

    wd_loss_weight = config["loss"]["wd_loss"]["weight"]
    wd_loss_function = WDLoss(
        num_group=num_group,
        mode=config["loss"]["wd_loss"]["mode"],
    )

    powd_loss_weight = config["loss"]["powd_loss"]["weight"]
    powd_loss_function = PredictionOversamplingWDLoss(
        num_group=num_group,
        mode=config["loss"]["powd_loss"]["mode"],
        lcm=config["loss"]["powd_loss"]["lcm"],
    )

    gapreg_loss_weight = config["loss"]["gapreg_loss"]["weight"]
    gapreg_loss_function = GapReg(
        num_group=num_group,
        mode=config["loss"]["gapreg_loss"]["mode"],
    )

    l2_loss_weight = config["loss"]["l2_loss"]["weight"]
    l2_loss_function = L2Loss(
        num_group=num_group,
        mode=config["loss"]["l2_loss"]["mode"],
    )

    loss_function = FairnessLoss(
        base_loss_weight=base_loss_weight,
        base_loss_function=base_loss_function,
        wd_loss_weight=wd_loss_weight,
        wd_loss_function=wd_loss_function,
        powd_loss_weight=powd_loss_weight,
        powd_loss_function=powd_loss_function,
        gapreg_loss_weight=gapreg_loss_weight,
        gapreg_loss_function=gapreg_loss_function,
        l2_loss_weight=l2_loss_weight,
        l2_loss_function=l2_loss_function,
    )
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
    binaryaccuracy_fn = BinaryAccuracy(threshold=tau)
    binaryaverageprecision_fn = BinaryAveragePrecision()
    dp_fn = DP(num_class=num_class, num_group=num_group, tau=tau)
    spdd_fn = SPDD(num_class=num_class, num_group=num_group, interval=interval)
    equal_opportunity_fn = Equal_Opportunity(
        num_class=num_class, num_group=num_group, tau=tau
    )
    equalized_odds_fn = Equalized_Odds(
        num_class=num_class, num_group=num_group, tau=tau
    )
    gs_fn = GS(num_class=num_class, num_group=num_group, tau=tau)
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
    train_steps = 0
    valid_steps = 0
    prev_acc = 0
    for epoch in range(epochs):
        ##################################################################################
        # training
        train_epoch_pred = []
        train_epoch_target = []
        train_epoch_group = []
        for image, target, group in tqdm(train_dataloader):
            image = image.to(device)
            target = target.type(torch.float32).to(device)
            group = group.type(torch.float32).to(device)

            pred, feature = model(image)

            loss_dict = loss_function(
                feature=feature,
                pred=pred,
                target=target,
                group=group,
            )

            for loss_name, loss_value in loss_dict.items():
                writer.add_scalar("train/" + loss_name, loss_value, train_steps)
            train_steps += 1

            optimizer.zero_grad()
            loss_dict["total_loss"].backward()
            optimizer.step()

            train_epoch_pred.append(pred.squeeze(-1).detach().cpu())
            train_epoch_target.append(target.squeeze(-1).detach().cpu())
            train_epoch_group.append(group.squeeze(-1).detach().cpu())
        ##################################################################################
        # train metrics
        train_epoch_pred = torch.concat(train_epoch_pred)
        train_epoch_target = torch.concat(train_epoch_target)
        train_epoch_group = torch.concat(train_epoch_group)

        train_epoch_target = torch.where(train_epoch_target >= tau, 1.0, 0.0)
        binaryaccuracy = binaryaccuracy_fn(train_epoch_pred, train_epoch_target)
        binaryaverageprecision = binaryaverageprecision_fn(
            train_epoch_pred, train_epoch_target.type(torch.int8)
        )
        dp_mean, dp_max = dp_fn(train_epoch_pred, train_epoch_target, train_epoch_group)
        spdd_mean, spdd_max = spdd_fn(
            train_epoch_pred, train_epoch_target, train_epoch_group
        )

        equal_opportunity_mean, equal_opportunity_max = equal_opportunity_fn(
            train_epoch_pred, train_epoch_target, train_epoch_group
        )
        equalized_odds_mean, equalized_odds_max = equalized_odds_fn(
            train_epoch_pred, train_epoch_target, train_epoch_group
        )
        gs_mean, gs_max = gs_fn(train_epoch_pred, train_epoch_target, train_epoch_group)

        print("=" * 100)
        print("training")
        print("epoch :", epoch)
        print("acc :", binaryaccuracy)
        print("ap :", binaryaverageprecision)
        print("dp :", dp_mean, dp_max)
        print("spdd :", spdd_mean, spdd_max)
        print("equal_opportunity :", equal_opportunity_mean, equal_opportunity_max)
        print("equalized_odds :", equalized_odds_mean, equalized_odds_max)
        print("gs :", gs_mean, gs_max)
        print("=" * 100)

        writer.add_scalar("train/binaryaccuracy", binaryaccuracy, epoch)
        writer.add_scalar("train/binaryaverageprecision", binaryaverageprecision, epoch)
        writer.add_scalar("train/Male/dp_mean", dp_mean, epoch)
        writer.add_scalar("train/Male/dp_max", dp_max, epoch)
        writer.add_scalar(
            "train/Male/equal_opportunity_mean", equal_opportunity_mean, epoch
        )
        writer.add_scalar(
            "train/Male/equal_opportunity_max", equal_opportunity_max, epoch
        )
        writer.add_scalar("train/Male/equalized_odds_mean", equalized_odds_mean, epoch)
        writer.add_scalar("train/Male/equalized_odds_max", equalized_odds_max, epoch)
        writer.add_scalar("train/Male/spdd_mean", spdd_mean, epoch)
        writer.add_scalar("train/Male/spdd_max", spdd_max, epoch)
        writer.add_scalar("train/Male/gs_mean", gs_mean, epoch)
        writer.add_scalar("train/Male/gs_max", gs_max, epoch)

        del train_epoch_pred
        del train_epoch_target
        del train_epoch_group
        ##################################################################################
        # model 저장
        # filepath = os.path.join("./", model_root, str(epoch).zfill(3) + ".pt")
        # print("filepath :", filepath)
        # torch.save(model, filepath)
        ##################################################################################
        # validation
        valid_epoch_pred = []
        valid_epoch_target = []
        valid_epoch_group = []
        with torch.no_grad():
            for image, target, group in tqdm(valid_dataloader):
                image = image.to(device)
                target = target.type(torch.float32).to(device)
                group = group.type(torch.float32).to(device)

                pred, feature = model(image)

                loss_dict = loss_function(
                    feature=feature,
                    pred=pred,
                    target=target,
                    group=group,
                )

                for loss_name, loss_value in loss_dict.items():
                    writer.add_scalar("valid/" + loss_name, loss_value, train_steps)
                valid_steps += 1

                valid_epoch_pred.append(pred.squeeze(-1).detach().cpu())
                valid_epoch_target.append(target.squeeze(-1).detach().cpu())
                valid_epoch_group.append(group.squeeze(-1).detach().cpu())
            ##################################################################################
            # train metrics
            valid_epoch_pred = torch.concat(valid_epoch_pred)
            valid_epoch_target = torch.concat(valid_epoch_target)
            valid_epoch_group = torch.concat(valid_epoch_group)

            valid_epoch_target = torch.where(valid_epoch_target >= tau, 1.0, 0.0)
            binaryaccuracy = binaryaccuracy_fn(valid_epoch_pred, valid_epoch_target)
            binaryaverageprecision = binaryaverageprecision_fn(
                valid_epoch_pred, valid_epoch_target.type(torch.int8)
            )
            dp_mean, dp_max = dp_fn(
                valid_epoch_pred, valid_epoch_target, valid_epoch_group
            )
            spdd_mean, spdd_max = spdd_fn(
                valid_epoch_pred, valid_epoch_target, valid_epoch_group
            )
            equal_opportunity_mean, equal_opportunity_max = equal_opportunity_fn(
                valid_epoch_pred, valid_epoch_target, valid_epoch_group
            )
            equalized_odds_mean, equalized_odds_max = equalized_odds_fn(
                valid_epoch_pred, valid_epoch_target, valid_epoch_group
            )
            gs_mean, gs_max = gs_fn(
                valid_epoch_pred, valid_epoch_target, valid_epoch_group
            )
            if prev_acc < binaryaccuracy.item():
                prev_acc = binaryaccuracy.item()
                filepath = os.path.join(model_path, "best_model.pt")
                print("filepath :", filepath)
                save_dict = {
                    "epoch": epoch,
                    "model": model,
                }
                torch.save(save_dict, filepath)

            print("=" * 100)
            print("validation")
            print("epoch :", epoch)
            print("acc :", binaryaccuracy)
            print("ap :", binaryaverageprecision)
            print("dp :", dp_mean, dp_max)
            print("spdd :", spdd_mean, spdd_max)
            print("equal_opportunity :", equal_opportunity_mean, equal_opportunity_max)
            print("equalized_odds :", equalized_odds_mean, equalized_odds_max)
            print("gs :", gs_mean, gs_max)
            print("=" * 100)

            writer.add_scalar("valid/binaryaccuracy", binaryaccuracy, epoch)
            writer.add_scalar(
                "valid/binaryaverageprecision", binaryaverageprecision, epoch
            )
            writer.add_scalar("valid/Male/dp_mean", dp_mean, epoch)
            writer.add_scalar("valid/Male/dp_max", dp_max, epoch)
            writer.add_scalar("valid/Male/spdd_mean", spdd_mean, epoch)
            writer.add_scalar("valid/Male/spdd_max", spdd_max, epoch)
            writer.add_scalar(
                "valid/Male/equal_opportunity_mean", equal_opportunity_mean, epoch
            )
            writer.add_scalar(
                "valid/Male/equal_opportunity_max", equal_opportunity_max, epoch
            )
            writer.add_scalar(
                "valid/Male/equalized_odds_mean", equalized_odds_mean, epoch
            )
            writer.add_scalar(
                "valid/Male/equalized_odds_max", equalized_odds_max, epoch
            )
            writer.add_scalar("valid/Male/gs_mean", gs_mean, epoch)
            writer.add_scalar("valid/Male/gs_max", gs_max, epoch)

        del valid_epoch_pred
        del valid_epoch_target
        del valid_epoch_group

    ##################################################################################
    # test
    test_transform = get_base_val_transform()
    test_dataset = CelebADataset_MSA(
        root=config["datamodule"]["image_dir"],
        split="test",
        transform=test_transform,
        gaussian_noise_transform=False,
        download=False,
        target_attribute=config["datamodule"]["target_attribute"],
        sensitive_attributes=config["datamodule"]["sensitive_attributes"],
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        num_workers=config["datamodule"]["num_workers"],
        batch_size=config["datamodule"]["batch_size"],
        persistent_workers=True,
    )
    filepath = os.path.join("./", model_path, "best_model.pt")
    save_dict = torch.load(filepath, map_location="cpu")
    epoch = save_dict["epoch"]
    model = save_dict["model"]
    model.to(device)
    ##################################################################################
    test_epoch_pred = []
    test_epoch_target = []
    test_epoch_group = []
    with torch.no_grad():
        for image, target, group in tqdm(test_dataloader):
            image = image.to(device)
            target = target.type(torch.float32).to(device)
            group = group.type(torch.float32).to(device)

            pred, _ = model(image)

            test_epoch_pred.append(pred.squeeze(-1).detach().cpu())
            test_epoch_target.append(target.squeeze(-1).detach().cpu())
            test_epoch_group.append(group.squeeze(-1).detach().cpu())
    ##################################################################################

    test_epoch_pred = torch.concat(test_epoch_pred)
    test_epoch_target = torch.concat(test_epoch_target)
    test_epoch_group = torch.concat(test_epoch_group)

    test_epoch_target = torch.where(test_epoch_target >= tau, 1.0, 0.0)
    binaryaccuracy = binaryaccuracy_fn(test_epoch_pred, test_epoch_target)
    binaryaverageprecision = binaryaverageprecision_fn(
        test_epoch_pred, test_epoch_target.type(torch.int8)
    )
    dp_mean, dp_max = dp_fn(test_epoch_pred, test_epoch_target, test_epoch_group)
    spdd_mean, spdd_max = spdd_fn(test_epoch_pred, test_epoch_target, test_epoch_group)
    equal_opportunity_mean, equal_opportunity_max = equal_opportunity_fn(
        test_epoch_pred, test_epoch_target, test_epoch_group
    )
    equalized_odds_mean, equalized_odds_max = equalized_odds_fn(
        test_epoch_pred, test_epoch_target, test_epoch_group
    )
    gs_mean, gs_max = gs_fn(test_epoch_pred, test_epoch_target, test_epoch_group)

    print("=" * 100)
    print("test")
    print("acc :", binaryaccuracy)
    print("ap :", binaryaverageprecision)
    print("dp :", dp_mean, dp_max)
    print("spdd :", spdd_mean, spdd_max)
    print("equal_opportunity :", equal_opportunity_mean, equal_opportunity_max)
    print("equalized_odds :", equalized_odds_mean, equalized_odds_max)
    print("gs :", gs_mean, gs_max)
    print("=" * 100)

    writer.add_scalar("test/binaryaccuracy", binaryaccuracy, epoch)
    writer.add_scalar("test/binaryaverageprecision", binaryaverageprecision, epoch)
    writer.add_scalar("test/Male/dp_mean", dp_mean, epoch)
    writer.add_scalar("test/Male/dp_max", dp_max, epoch)
    writer.add_scalar("test/Male/spdd_mean", spdd_mean, epoch)
    writer.add_scalar("test/Male/spdd_max", spdd_max, epoch)
    writer.add_scalar("test/Male/equal_opportunity_mean", equal_opportunity_mean, epoch)
    writer.add_scalar("test/Male/equal_opportunity_max", equal_opportunity_max, epoch)
    writer.add_scalar("test/Male/equalized_odds_mean", equalized_odds_mean, epoch)
    writer.add_scalar("test/Male/equalized_odds_max", equalized_odds_max, epoch)
    writer.add_scalar("test/Male/gs_mean", gs_mean, epoch)
    writer.add_scalar("test/Male/gs_max", gs_max, epoch)

    writer.flush()
    writer.close()
    results = {
        "best_epoch": epoch,
        "acc": round(binaryaccuracy.item(), 4),
        "ap": round(binaryaverageprecision.item(), 4),
        "dp_mean": round(dp_mean.item(), 4),
        "dp_max": round(dp_max.item(), 4),
        "equal_opportunity_mean": round(equal_opportunity_mean.item(), 4),
        "equal_opportunity_max": round(equal_opportunity_max.item(), 4),
        "equalized_odds_mean": round(equalized_odds_mean.item(), 4),
        "equalized_odds_max": round(equalized_odds_max.item(), 4),
        "gs_mean": round(gs_mean.item(), 4),
        "gs_max": round(gs_max.item(), 4),
    }

    filepath = os.path.join("./", logger_path, "test_results.json")
    with open(filepath, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    config_path = "./config/vanilla.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    main(config)
    config["trainer"]["learning_rate"] = 1.0e-04
    main(config)
    config["trainer"]["learning_rate"] = 1.0e-05
    main(config)
    config["trainer"]["learning_rate"] = 1.0e-06
    main(config)
