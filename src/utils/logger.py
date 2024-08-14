import os
import yaml
from torchmetrics import metric
from torch.utils.tensorboard import SummaryWriter


def set_loggers_folder(
    config: dict,
    logger_root: str,
    logger_name: str,
):
    try:
        logger_path = os.path.join(logger_root, logger_name)
        if len(os.listdir(logger_path)) == 0:
            logger_path = logger_path + "/version_0"
            os.makedirs(logger_path)
        else:
            number = str(int(sorted(os.listdir(logger_path))[-1][-1]) + 1)
            logger_path = logger_path + "/version_" + number
            os.makedirs(logger_path)
    except:
        logger_path = logger_path + "/version_0"
        os.makedirs(logger_path)

    print(logger_path)
    writer = SummaryWriter(logger_path)
    model_path = os.path.join(logger_path, "model")
    os.makedirs(model_path)
    print(model_path)

    config_path = os.path.join(logger_path, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    return {
        "logger_path": logger_path,
        "model_path": model_path,
        "writer": writer,
    }


def log_tensorboard(
    writer: SummaryWriter,
    performance_metrics: metric,
    fairness_metrics: metric,
    epoch: int,
    s: str = "valid",
):
    print(s + " epoch :", epoch)
    if performance_metrics != None:
        for metric in performance_metrics:
            key = metric._get_name()
            value = metric.compute()
            metric.reset()
            value = round(value.item(), 3)
            print(s + "/" + key, ":", value)
            writer.add_scalar(s + "/" + key, value, epoch)
            if key == performance_metrics[0]._get_name():
                accuracy = value

    if fairness_metrics != None:
        for metric in fairness_metrics:
            key = metric._get_name()
            value_mean, value_max = metric.compute()
            metric.reset()
            value_mean = round(value_mean.item(), 3)
            value_max = round(value_max.item(), 3)
            print(s + "/" + key + "_mean", ":", value_mean)
            print(s + "/" + key + "_max", ":", value_max)
            writer.add_scalar(s + "/" + key + "_mean", value_mean, epoch)
            writer.add_scalar(s + "/" + key + "_max", value_max, epoch)

    return accuracy
