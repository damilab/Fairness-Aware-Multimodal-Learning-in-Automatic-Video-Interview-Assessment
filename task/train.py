from dataset.transform import (
    get_base_train_transform,
    get_base_val_transform,
)
from dataset import CelebADataset_MSA
from torch.utils.data import DataLoader
from network import ResNet
import torch
from loss import PredictionOversamplingWDLoss
from torchmetrics.classification import BinaryAccuracy
from metric import DP, SPDD, GS
from metric import Equal_Opportunity, Equalized_Odds
import os
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import json


def main(config: dict):
    train_transform = get_base_train_transform()
    valid_transform = get_base_val_transform()
    sensitive_attributes = ["Male", "Young"]
    train_dataset = CelebADataset_MSA(
        root=".././data",
        split="train",
        transform=train_transform,
        gaussian_noise_transform=config["gaussian_noise_transform_in_train_dataset"],
        download=False,
        sensitive_attributes=sensitive_attributes,
    )
    valid_dataset = CelebADataset_MSA(
        root=".././data",
        split="valid",
        transform=valid_transform,
        gaussian_noise_transform=False,
        download=False,
        sensitive_attributes=sensitive_attributes,
    )
    num_workers = 4
    batch_size = 128
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        batch_size=batch_size,
        persistent_workers=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        batch_size=batch_size,
        persistent_workers=True,
    )

    model = ResNet(
        num_class=1,
        type="res18",
        pretrained_path="None",
    )

    epochs = config["epochs"]
    num_group = 4
    device = config["device"]
    bce_loss_fn = torch.nn.BCELoss()
    wd_loss_weight = config["wd_loss_lambda"]
    wd_loss_fn = PredictionOversamplingWDLoss(
        num_group=num_group, mode=config["wd_loss_mode"], lcm=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    num_class = 2
    tau = 0.5
    interval = 0.02
    binaryaccuracy_fn = BinaryAccuracy(threshold=tau)
    dp_fn = DP(num_class=num_class, num_group=num_group, tau=tau)
    spdd_fn = SPDD(num_class=num_class, num_group=num_group, interval=interval)
    equal_opportunity_fn = Equal_Opportunity(
        num_class=num_class, num_group=num_group, tau=tau
    )
    equalized_odds_fn = Equalized_Odds(
        num_class=num_class, num_group=num_group, tau=tau
    )
    gs_fn = GS(num_class=num_class, num_group=num_group, tau=tau)
    logger_root = config["logger_root"]
    logger_path = os.path.join("runs/", logger_root)
    print(logger_path)
    writer = SummaryWriter(logger_path)

    model_root = os.path.join(logger_path, "model")
    print(model_root)
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    model.to(device)
    train_steps = 0
    valid_steps = 0
    for epoch in range(epochs):
        ##################################################################################
        # training
        train_epoch_bce_loss = []
        train_epoch_wd_loss = []

        train_epoch_pred = []
        train_epoch_target = []
        train_epoch_group = []
        for image, target, group in tqdm(train_dataloader):
            image = image.to(device)
            target = target.type(torch.float32).to(device)
            group = group.type(torch.float32).to(device)

            pred, _ = model(image)

            bce_loss = bce_loss_fn(pred, target)
            # wd_loss = wd_loss_fn(pred, group)
            # loss = bce_loss + (wd_loss_weight * wd_loss)
            loss = bce_loss

            writer.add_scalar("train/bce_loss", bce_loss, train_steps)
            # writer.add_scalar("train/wd_loss", wd_loss, train_steps)
            train_steps += 1

            optimizer.zero_grad()
            loss.backward()
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
        print("dp :", dp_mean, dp_max)
        print("spdd :", spdd_mean, spdd_max)
        print("equal_opportunity :", equal_opportunity_mean, equal_opportunity_max)
        print("equalized_odds :", equalized_odds_mean, equalized_odds_max)
        print("gs :", gs_mean, gs_max)
        print("=" * 100)

        writer.add_scalar("train/binaryaccuracy", binaryaccuracy, epoch)
        writer.add_scalar("train/gender_age/dp_mean", dp_mean, epoch)
        writer.add_scalar("train/gender_age/dp_max", dp_max, epoch)
        writer.add_scalar(
            "train/gender_age/equal_opportunity_mean", equal_opportunity_mean, epoch
        )
        writer.add_scalar(
            "train/gender_age/equal_opportunity_max", equal_opportunity_max, epoch
        )
        writer.add_scalar(
            "train/gender_age/equalized_odds_mean", equalized_odds_mean, epoch
        )
        writer.add_scalar(
            "train/gender_age/equalized_odds_max", equalized_odds_max, epoch
        )
        writer.add_scalar("train/gender_age/spdd_mean", spdd_mean, epoch)
        writer.add_scalar("train/gender_age/spdd_max", spdd_max, epoch)
        writer.add_scalar("train/gender_age/gs_mean", gs_mean, epoch)
        writer.add_scalar("train/gender_age/gs_max", gs_max, epoch)

        del train_epoch_pred
        del train_epoch_target
        del train_epoch_group
        ##################################################################################
        # model 저장
        filepath = os.path.join("./", model_root, str(epoch).zfill(3) + ".pt")
        print("filepath :", filepath)
        torch.save(model, filepath)
        ##################################################################################
        # validation
        prev_acc = 0
        valid_epoch_pred = []
        valid_epoch_target = []
        valid_epoch_group = []
        with torch.no_grad():
            for image, target, group in tqdm(valid_dataloader):
                image = image.to(device)
                target = target.type(torch.float32).to(device)
                group = group.type(torch.float32).to(device)

                pred, _ = model(image)

                bce_loss = bce_loss_fn(pred, target)
                # wd_loss = wd_loss_fn(pred, group)
                # loss = bce_loss + (wd_loss_weight * wd_loss)

                writer.add_scalar("valid/bce_loss", bce_loss, valid_steps)
                # writer.add_scalar("valid/wd_loss", wd_loss, valid_steps)
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
            if prev_acc < binaryaccuracy:
                filepath = os.path.join("./", model_root, "best_model.pt")
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
            print("dp :", dp_mean, dp_max)
            print("spdd :", spdd_mean, spdd_max)
            print("equal_opportunity :", equal_opportunity_mean, equal_opportunity_max)
            print("equalized_odds :", equalized_odds_mean, equalized_odds_max)
            print("gs :", gs_mean, gs_max)
            print("=" * 100)

            writer.add_scalar("valid/binaryaccuracy", binaryaccuracy, epoch)
            writer.add_scalar("valid/gender_age/dp_mean", dp_mean, epoch)
            writer.add_scalar("valid/gender_age/dp_max", dp_max, epoch)
            writer.add_scalar("valid/gender_age/spdd_mean", spdd_mean, epoch)
            writer.add_scalar("valid/gender_age/spdd_max", spdd_max, epoch)
            writer.add_scalar(
                "valid/gender_age/equal_opportunity_mean", equal_opportunity_mean, epoch
            )
            writer.add_scalar(
                "valid/gender_age/equal_opportunity_max", equal_opportunity_max, epoch
            )
            writer.add_scalar(
                "valid/gender_age/equalized_odds_mean", equalized_odds_mean, epoch
            )
            writer.add_scalar(
                "valid/gender_age/equalized_odds_max", equalized_odds_max, epoch
            )
            writer.add_scalar("valid/gender_age/gs_mean", gs_mean, epoch)
            writer.add_scalar("valid/gender_age/gs_max", gs_max, epoch)

        del valid_epoch_pred
        del valid_epoch_target
        del valid_epoch_group

    ##################################################################################
    # test
    test_transform = get_base_val_transform()
    sensitive_attributes = ["Male", "Young"]
    test_dataset = CelebADataset_MSA(
        root=".././data",
        split="test",
        transform=test_transform,
        gaussian_noise_transform=False,
        download=False,
        sensitive_attributes=sensitive_attributes,
    )
    num_workers = 8
    batch_size = 256
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        batch_size=batch_size,
        persistent_workers=True,
    )
    filepath = os.path.join("./", model_root, "best_model.pt")
    model = torch.load(filepath, map_location="cpu")
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

    binaryaccuracy = binaryaccuracy_fn(test_epoch_pred, test_epoch_target)
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
    print("dp :", dp_mean, dp_max)
    print("spdd :", spdd_mean, spdd_max)
    print("equal_opportunity :", equal_opportunity_mean, equal_opportunity_max)
    print("equalized_odds :", equalized_odds_mean, equalized_odds_max)
    print("gs :", gs_mean, gs_max)
    print("=" * 100)

    writer.add_scalar("test/binaryaccuracy", binaryaccuracy, epoch)
    writer.add_scalar("test/gender_age/dp_mean", dp_mean, epoch)
    writer.add_scalar("test/gender_age/dp_max", dp_max, epoch)
    writer.add_scalar("test/gender_age/spdd_mean", spdd_mean, epoch)
    writer.add_scalar("test/gender_age/spdd_max", spdd_max, epoch)
    writer.add_scalar(
        "test/gender_age/equal_opportunity_mean", equal_opportunity_mean, epoch
    )
    writer.add_scalar(
        "test/gender_age/equal_opportunity_max", equal_opportunity_max, epoch
    )
    writer.add_scalar("test/gender_age/equalized_odds_mean", equalized_odds_mean, epoch)
    writer.add_scalar("test/gender_age/equalized_odds_max", equalized_odds_max, epoch)
    writer.add_scalar("test/gender_age/gs_mean", gs_mean, epoch)
    writer.add_scalar("test/gender_age/gs_max", gs_max, epoch)

    writer.flush()
    writer.close()
    results = {
        "best_epoch": epoch,
        "acc": round(binaryaccuracy.item(), 4),
        "dp_mean": round(dp_mean.item(), 4),
        "dp_max": round(dp_max.item(), 4),
        "equal_opportunity_mean": round(equal_opportunity_mean.item(), 4),
        "equal_opportunity_max": round(equal_opportunity_max.item(), 4),
        "equalized_odds_mean": round(equalized_odds_mean.item(), 4),
        "equalized_odds_max": round(equalized_odds_max.item(), 4),
        "gs_mean": round(gs_mean.item(), 4),
        "gs_max": round(gs_max.item(), 4),
    }

    filepath = os.path.join("./", model_root, "test_results.json")
    with open(filepath, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    # config = {
    #     "epochs": 200,
    #     "device": 0,
    #     "logger_root": "vanilla/version_0",
    #     "gaussian_noise_transform_in_train_dataset": False,
    #     "learning_rate": 1.0e-04,
    #     "wd_loss_mode": "mean",
    #     "wd_loss_lambda": 0,
    # }
    # main(config)

    config = {
        "epochs": 200,
        "device": 0,
        "logger_root": "gaussian_noise/version_0",
        "learning_rate": 1.0e-04,
        "gaussian_noise_transform_in_train_dataset": True,
        "wd_loss_mode": "mean",
        "wd_loss_lambda": 0,
    }
    main(config)
