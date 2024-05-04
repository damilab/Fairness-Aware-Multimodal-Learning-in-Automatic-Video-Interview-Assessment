import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import yaml

from dataset.transform import get_base_val_transform
from dataset import CelebADataset_MSA
from torch.utils.data import DataLoader
from network import ResNet

import torch

from torchmetrics.classification import BinaryAccuracy, BinaryAveragePrecision
from metric import DP, SPDD, GS
from metric import Equal_Opportunity, Equalized_Odds

from tqdm import tqdm
import json


def main(config: dict):
    #############################################################
    # test dataset & dataloader
    test_transform = get_base_val_transform()
    test_dataset = CelebADataset_MSA(
        root=config["datamodule"]["image_dir"],
        split="test",
        transform=test_transform,
        gaussian_noise_transform=False,
        download=False,
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
    #############################################################
    # metrics
    num_group = 2 ** len(config["datamodule"]["sensitive_attributes"])
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
    #############################################################
    # model
    device = config["trainer"]["device"]
    filepath = os.path.join("./", model_path, "best_model.pt")
    save_dict = torch.load(filepath, map_location="cpu")
    epoch = save_dict["epoch"]
    model = save_dict["model"]
    model.to(device)
    #############################################################
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
    logger_path = config["logger_path"]
    filepath = os.path.join("./", logger_path, "test_results.json")
    with open(filepath, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    logger_path = "./runs/l2/version_2/"
    config_path = os.path.join(logger_path, "config.yaml")
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    config["logger_path"] = logger_path
    model_path = os.path.join(config_path, "model/best_model.pt")
    config["model_path"] = model_path
    main(config)
