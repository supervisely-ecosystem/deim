import os

from dotenv import load_dotenv

import supervisely as sly
from supervisely.nn.utils import ModelSource, RuntimeType

load_dotenv(os.path.expanduser("~/supervisely.env"))
load_dotenv("local.env")

api: sly.Api = sly.Api.from_env()

task_id = 68910  # <---- Change this to your task_id
method = "deploy_from_api"


# Pretrained
pretrained_model_data = {
    "deploy_params": {
        "model_files": {
            "config": "deim_r18vd_120e_coco.yml",
            "checkpoint": "https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_rtdetrv2_r18vd_coco_120e.pth",
        },
        "model_source": ModelSource.PRETRAINED,
        "model_info": {
            "Model": "DEIM RT-DETRv2-S",
            "dataset": "COCO",
            "AP_val": 49.0,
            "Params(M)": 20,
            "Latency": "4.59ms",
            "GFLOPs": 60,
            "meta": {
                "task_type": "object detection",
                "model_name": "DEIM RT-DETRv2-S",
                "model_files": {
                    "checkpoint": "https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_rtdetrv2_r18vd_coco_120e.pth",
                    "config": "deim_r18vd_120e_coco.yml"
                },
            },
        },
        "device": "cuda",
        "runtime": RuntimeType.PYTORCH,
    }
}

# Custom
# @TODO: add deploy_params.json to team files for convenience?
custom_model_data = {
    "deploy_params": {
        "model_files": {
            "config": "/experiments/43192_Apples/75102_DEIM/model_config.yml",
            "checkpoint": "/experiments/43192_Apples/75102_DEIM/checkpoints/best.pth",
        },
        "model_source": ModelSource.CUSTOM,
        "model_info": {
            "artifacts_dir": "/experiments/43192_Apples/75102_DEIM",
            "framework_name": "DEIM",
            "model_name": "DEIM RT-DETRv2-S",
            "model_meta": "model_meta.json",
        },
        "device": "cuda",
        "runtime": RuntimeType.PYTORCH,
    }
}

api.app.send_request(task_id, method, custom_model_data)
