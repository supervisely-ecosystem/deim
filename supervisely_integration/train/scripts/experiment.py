import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely.template.experiment.experiment_generator import ExperimentGenerator
from supervisely_integration.serve.main import DEIM

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

team_id = sly.env.team_id()
workspace_id = sly.env.workspace_id()

experiment_info = {
    "experiment_name": "14748_Animals (W)_DEIM D-FINE-N",
    "framework_name": "DEIM",
    "model_name": "DEIM D-FINE-N",
    "task_type": "object detection",
    "project_id": 2067,
    "task_id": 14748,
    "model_files": {"config": "model_config.yml"},
    "checkpoints": [
        "checkpoints/best.pth",
        "checkpoints/checkpoint0024.pth",
        "checkpoints/checkpoint0049.pth",
        "checkpoints/last.pth",
    ],
    "best_checkpoint": "best.pth",
    "export": {},
    "app_state": "app_state.json",
    "model_meta": "model_meta.json",
    "hyperparameters": "hyperparameters.yaml",
    "artifacts_dir": "/experiments/2067_Animals (W)/14748_DEIM/",
    "datetime": "2025-05-30 09:03:01",
    "evaluation_report_id": 552019,
    "evaluation_report_link": "https://dev.internal.supervisely.com/model-benchmark?id=552019",
    "evaluation_metrics": {
        "mAP": 0.42239271909245313,
        "AP50": 0.5211519394855173,
        "AP75": 0.47561756927831,
        "f1": 0.2870967741935484,
        "precision": 0.1816326530612245,
        "recall": 0.13692307692307693,
        "iou": 0.8793985236012154,
        "classification_accuracy": 1,
        "calibration_score": 0.8622353937598518,
        "f1_optimal_conf": 0.35821080207824707,
        "expected_calibration_error": 0.1377646062401482,
        "maximum_calibration_error": 0.26885319418377346,
    },
    "logs": {"type": "tensorboard", "link": "/experiments/2067_Animals (W)/14748_DEIM/logs/"},
    "train_val_split": "train_val_split.json",
    "train_size": 27,
    "val_size": 27,
}

model_meta = {
    "classes": [
        {
            "title": "cat",
            "description": "",
            "shape": "rectangle",
            "color": "#A80B10",
            "geometry_config": {},
            "id": 52102,
            "hotkey": "",
        },
        {
            "title": "dog",
            "description": "",
            "shape": "rectangle",
            "color": "#B8E986",
            "geometry_config": {},
            "id": 52103,
            "hotkey": "",
        },
        {
            "title": "horse",
            "description": "",
            "shape": "rectangle",
            "color": "#9F21DE",
            "geometry_config": {},
            "id": 52104,
            "hotkey": "",
        },
        {
            "title": "sheep",
            "description": "",
            "shape": "rectangle",
            "color": "#1EA49B",
            "geometry_config": {},
            "id": 52105,
            "hotkey": "",
        },
        {
            "title": "squirrel",
            "description": "",
            "shape": "rectangle",
            "color": "#F8E71C",
            "geometry_config": {},
            "id": 52106,
            "hotkey": "",
        },
    ],
    "tags": [
        {
            "name": "animal age group",
            "value_type": "oneof_string",
            "color": "#F5A623",
            "values": ["juvenile", "adult", "senior"],
            "id": 7686,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "animal age group_1",
            "value_type": "any_string",
            "color": "#8A0F59",
            "id": 7687,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "animal count",
            "value_type": "any_number",
            "color": "#E3BE1C",
            "id": 7688,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "cat",
            "value_type": "none",
            "color": "#A80B10",
            "id": 7689,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "dog",
            "value_type": "none",
            "color": "#B8E986",
            "id": 7690,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "horse",
            "value_type": "none",
            "color": "#9F21DE",
            "id": 7691,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "imgtag",
            "value_type": "none",
            "color": "#FF03D6",
            "id": 7692,
            "hotkey": "",
            "applicable_type": "imagesOnly",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "sheep",
            "value_type": "none",
            "color": "#1EA49B",
            "id": 7693,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
        {
            "name": "squirrel",
            "value_type": "none",
            "color": "#F8E71C",
            "id": 7694,
            "hotkey": "",
            "applicable_type": "all",
            "classes": [],
            "target_type": "all",
        },
    ],
    "projectType": "images",
    "projectSettings": {
        "multiView": {"enabled": False, "tagName": None, "tagId": None, "isSynced": False}
    },
}
model_meta = sly.ProjectMeta.from_json(model_meta)

hyperparameters_yaml = """
epoches: 50
batch_size: 16
eval_spatial_size: [640, 640]  # height, width

checkpoint_freq: 25  # set 0 to keep only best and last checkpoints
save_optimizer: false
save_ema: false

optimizer:
  type: AdamW
  # lr: 0.0002
  betas: [0.9, 0.999]
  weight_decay: 0.0001

clip_max_norm: 1.0  # gradient clipping

## DEIM LR-Scheduler
lrsheduler: flatcosine
warmup_iter: 20  # first iterations with linear warmup
flat_epoch: 29    # epoches with constant learning rate
no_aug_epoch: 0   # epoches without augmentation

use_ema: False  # Exponential Moving Average for model weights
ema:
  type: ModelEMA
  decay: 0.9999
  warmups: 400

use_amp: True  # Automatic Mixed Precision
"""

app_options = {
    "demo": {
        "path": "supervisely_integration/demo",
    },
}


experiment = ExperimentGenerator(
    api=api,
    experiment_info=experiment_info,
    hyperparameters=hyperparameters_yaml,
    model_meta=model_meta,
    serving_class=DEIM,
    team_id=team_id,
    output_dir="./experiment_report",
    app_options=app_options,
)

experiment.generate()
experiment.upload_to_artifacts()
