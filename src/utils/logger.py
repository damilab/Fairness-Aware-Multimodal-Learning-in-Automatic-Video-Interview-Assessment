import os
import yaml
from torch.utils.tensorboard import SummaryWriter


def set_loggers_folder(
    config: dict,
    logger_root: str,
    logger_name: str,
):
    logger_path = os.path.join(logger_root, logger_name)
    try:
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
