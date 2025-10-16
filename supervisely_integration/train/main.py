import os
import shutil
from multiprocessing import cpu_count

import supervisely as sly
import torch
import yaml
from supervisely.nn import ModelSource, RuntimeType
from supervisely.nn.training.train_app import TrainApp

from engine.core import YAMLConfig
from engine.solver import DetSolver
from supervisely_integration.export import export_onnx, export_tensorrt
from supervisely_integration.serve.deim import DEIM

base_path = "supervisely_integration/train"
train = TrainApp(
    "DEIM",
    f"supervisely_integration/models.json",
    f"{base_path}/hyperparameters.yaml",
    f"{base_path}/app_options.yaml",
)

inference_settings = "supervisely_integration/serve/inference_settings.yaml"
train.register_inference_class(DEIM, inference_settings)


@train.start
def start_training():
    train_ann_path, val_ann_path = convert_data()
    checkpoint = train.model_files["checkpoint"]
    custom_config_path = prepare_config(train_ann_path, val_ann_path)
    cfg = YAMLConfig(
        custom_config_path,
        tuning=checkpoint,
    )
    _set_input_size_dataloaders(cfg.yaml_cfg, list(cfg.yaml_cfg["eval_spatial_size"]))
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # dump resolved config
    model_config_path = f"{output_dir}/model_config.yml"
    with open(model_config_path, "w") as f:
        yaml.dump(cfg.yaml_cfg, f)
    remove_include(model_config_path)
    # train
    tensorboard_logs = f"{output_dir}/summary"
    train.start_tensorboard(tensorboard_logs)
    solver = DetSolver(cfg)
    solver.fit()
    # gather experiment info
    experiment_info = {
        "model_name": train.model_name,
        "model_files": {"config": model_config_path},
        "checkpoints": output_dir,
        "best_checkpoint": "best.pth",
    }
    return experiment_info


@train.export_onnx
def to_onnx(experiment_info: dict):
    export_path = export_onnx(
        experiment_info["best_checkpoint"],
        experiment_info["model_files"]["config"],
    )
    return export_path


@train.export_tensorrt
def to_tensorrt(experiment_info: dict):
    onnx_path = export_onnx(
        experiment_info["best_checkpoint"],
        experiment_info["model_files"]["config"],
    )
    export_path = export_tensorrt(
        onnx_path,
        fp16=True,
    )
    return export_path


def convert_data():
    project = train.sly_project
    meta = project.meta

    train_dataset: sly.Dataset = project.datasets.get("train")
    train_ann_path = train_dataset.to_coco(meta, dest_dir=train_dataset.directory)

    val_dataset: sly.Dataset = project.datasets.get("val")
    val_ann_path = val_dataset.to_coco(meta, dest_dir=val_dataset.directory)
    return train_ann_path, val_ann_path


def prepare_config(train_ann_path: str, val_ann_path: str):
    if train.model_name.startswith("DEIM D-FINE"):
        deim_config_dir = "configs/deim_dfine"
    elif train.model_name.startswith("DEIM RT-DETRv2"):
        deim_config_dir = "configs/deim_rtdetrv2"
    else:
        deim_config_dir = "configs/deimv2"
    if train.model_source == ModelSource.CUSTOM:
        config_path = train.model_files["config"]
        config = os.path.basename(config_path)
        shutil.move(config_path, f"{deim_config_dir}/{config}")
    else:
        config = train.model_files["config"]

    custom_config = train.hyperparameters
    custom_config["__include__"] = [config]
    custom_config["remap_mscoco_category"] = train.num_classes <= 80
    custom_config["num_classes"] = train.num_classes
    custom_config["print_freq"] = 50

    custom_config.setdefault("train_dataloader", {}).setdefault("dataset", {})
    custom_config["train_dataloader"]["dataset"]["img_folder"] = f"{train.train_dataset_dir}/img"
    custom_config["train_dataloader"]["dataset"]["ann_file"] = train_ann_path

    custom_config.setdefault("val_dataloader", {}).setdefault("dataset", {})
    custom_config["val_dataloader"]["dataset"]["img_folder"] = f"{train.val_dataset_dir}/img"
    custom_config["val_dataloader"]["dataset"]["ann_file"] = val_ann_path

    if "batch_size" in custom_config:
        batch_size = custom_config["batch_size"]
        num_workers = min(batch_size, 8, cpu_count())
        custom_config["train_dataloader"]["total_batch_size"] = batch_size
        custom_config["val_dataloader"]["total_batch_size"] = batch_size * 2
        custom_config["train_dataloader"]["num_workers"] = num_workers
        custom_config["val_dataloader"]["num_workers"] = num_workers

    custom_config_path = f"{deim_config_dir}/custom_config.yml"
    with open(custom_config_path, "w") as f:
        yaml.dump(custom_config, f)

    return custom_config_path


def remove_include(config_path: str):
    # del "__include__" and rewrite the config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "__include__" in config:
        config.pop("__include__")
        with open(config_path, "w") as f:
            yaml.dump(config, f)


def _set_input_size_dataloaders(custom_config: dict, size: list):
    for dataloader in ["train_dataloader", "val_dataloader"]:
        ops = custom_config[dataloader]["dataset"]["transforms"]["ops"]
        for i, op in enumerate(ops):
            if op["type"] == "Resize":
                ops[i]["size"] = list(size)
                break
    custom_config["train_dataloader"]["collate_fn"]["base_size"] = list(size)


if train.auto_start:
    train.start_in_thread()
