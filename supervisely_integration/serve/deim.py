import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torchvision.transforms import ToTensor

import supervisely as sly
from engine.core import YAMLConfig
from engine.data.dataset.coco_dataset import mscoco_category2name
from supervisely.io.fs import get_file_name_with_ext
from supervisely.nn.inference import CheckpointInfo, ModelSource, RuntimeType, Timer
from supervisely.nn.prediction_dto import PredictionBBox
from supervisely_integration.export import export_onnx, export_tensorrt

SERVE_PATH = "supervisely_integration/serve"
CONFIG_DIR = "configs"


class DEIM(sly.nn.inference.ObjectDetection):
    FRAMEWORK_NAME = "DEIM"
    MODELS = "supervisely_integration/models.json"
    APP_OPTIONS = f"{SERVE_PATH}/app_options.yaml"
    INFERENCE_SETTINGS = f"{SERVE_PATH}/inference_settings.yaml"
    _dynamic_input_size = False

    def load_model(
        self, model_files: dict, model_info: dict, model_source: str, device: str, runtime: str
    ):
        self._clear_global_config()
        if model_source == ModelSource.CUSTOM:
            checkpoint_path, config_path = self._prepare_custom_model(model_files)
        else:
            checkpoint_path, config_path = self._prepare_pretrained_model(model_files, model_info)

        self._load_transforms(config_path)
        if runtime == RuntimeType.PYTORCH:
            self._load_pytorch(checkpoint_path, config_path, device)
        elif runtime == RuntimeType.ONNXRUNTIME:
            self._load_onnx(checkpoint_path, device)
        elif runtime == RuntimeType.TENSORRT:
            self._load_tensorrt(checkpoint_path, device)

    def predict_benchmark(self, images_np: List[np.ndarray], settings: dict = None):
        if self.runtime == RuntimeType.PYTORCH:
            return self._predict_pytorch(images_np, settings)
        elif self.runtime == RuntimeType.ONNXRUNTIME:
            return self._predict_onnx(images_np, settings)
        elif self.runtime == RuntimeType.TENSORRT:
            return self._predict_tensorrt(images_np, settings)

    # Loaders ------------------ #
    def _load_transforms(self, config_path: str):
        with open(config_path, "r") as f:
            yaml_cfg = yaml.safe_load(f)

        # Use default spatial size if eval_spatial_size is not present
        spatial_size = yaml_cfg.get("eval_spatial_size", [640, 640])
        h, w = spatial_size
        self.input_size = [w, h]
        self.transforms = T.Compose([T.Resize((h, w)), T.ToTensor()])

    def _load_pytorch(self, checkpoint_path: str, config_path: str, device: str):
        self.cfg = YAMLConfig(config_path, resume=checkpoint_path)
        if "HGNetv2" in self.cfg.yaml_cfg:
            self.cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        self.model = self.cfg.model
        self.model.load_state_dict(state, strict=False)
        self.model.deploy().to(device)
        self.postprocessor = self.cfg.postprocessor.deploy().to(device)

    def _load_onnx(self, onnx_path: str, device: str):
        import onnxruntime

        providers = ["CUDAExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
        if device != "cpu":
            assert onnxruntime.get_device() == "GPU", "ONNXRuntime is not configured to use GPU"
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=providers)

    def _load_tensorrt(self, engine_path: str, device: str):
        from tools.inference.trt_inf import TRTInference

        assert device != "cpu", "TensorRT is not supported on CPU"
        self.engine = TRTInference(engine_path, device)
        self.max_batch_size = 1

    # -------------------------- #

    # Predictions -------------- #
    @torch.no_grad()
    def _predict_pytorch(
        self, images_np: List[np.ndarray], settings: dict = None
    ) -> Tuple[List[List[PredictionBBox]], dict]:
        # 1. Preprocess
        with Timer() as preprocess_timer:
            img_input, size_input, orig_target_sizes = self._prepare_input(images_np)
        # 2. Inference
        with Timer() as inference_timer:
            outputs = self.model(img_input)
        # 3. Postprocess
        with Timer() as postprocess_timer:
            labels, boxes, scores = self.postprocessor(outputs, orig_target_sizes)
            labels, boxes, scores = labels.cpu().numpy(), boxes.cpu().numpy(), scores.cpu().numpy()
            predictions = self._format_predictions(labels, boxes, scores, settings)
        benchmark = {
            "preprocess": preprocess_timer.get_time(),
            "inference": inference_timer.get_time(),
            "postprocess": postprocess_timer.get_time(),
        }
        return predictions, benchmark

    def _predict_onnx(
        self, images_np: List[np.ndarray], settings: dict
    ) -> Tuple[List[List[PredictionBBox]], dict]:
        # 1. Preprocess
        with Timer() as preprocess_timer:
            img_input, size_input, orig_sizes = self._prepare_input(images_np, device="cpu")
            img_input, orig_sizes = img_input.data.numpy(), orig_sizes.data.numpy()
        # 2. Inference
        with Timer() as inference_timer:
            labels, boxes, scores = self.onnx_session.run(
                output_names=None,
                input_feed={"images": img_input, "orig_target_sizes": orig_sizes},
            )
        # 3. Postprocess
        with Timer() as postprocess_timer:
            predictions = self._format_predictions(labels, boxes, scores, settings)
        benchmark = {
            "preprocess": preprocess_timer.get_time(),
            "inference": inference_timer.get_time(),
            "postprocess": postprocess_timer.get_time(),
        }
        return predictions, benchmark

    @torch.no_grad()
    def _predict_tensorrt(self, images_np: List[np.ndarray], settings: dict):
        # 1. Preprocess
        with Timer() as preprocess_timer:
            img_input, size_input, orig_sizes = self._prepare_input(images_np)
        # 2. Inference
        with Timer() as inference_timer:
            output = self.engine({"images": img_input, "orig_target_sizes": orig_sizes})
            labels = output["labels"].cpu().numpy()
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
        # 3. Postprocess
        with Timer() as postprocess_timer:
            predictions = self._format_predictions(labels, boxes, scores, settings)
        benchmark = {
            "preprocess": preprocess_timer.get_time(),
            "inference": inference_timer.get_time(),
            "postprocess": postprocess_timer.get_time(),
        }
        return predictions, benchmark

    def _prepare_input(self, images_np: List[np.ndarray], device=None):
        if device is None:
            device = self.device
        imgs_pil = [Image.fromarray(img) for img in images_np]
        orig_sizes = torch.as_tensor([img.size for img in imgs_pil])
        img_input = torch.stack([self.transforms(img) for img in imgs_pil])
        size_input = torch.tensor([self.input_size * len(images_np)]).reshape(-1, 2)
        return img_input.to(device), size_input.to(device), orig_sizes.to(device)

    def _format_prediction(
        self, labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, conf_tresh: float
    ) -> List[PredictionBBox]:
        predictions = []
        for label, bbox_xyxy, score in zip(labels, boxes, scores):
            if score < conf_tresh:
                continue
            class_name = self.classes[label]

            bbox_xyxy = np.round(bbox_xyxy).astype(float)
            bbox_xyxy = np.clip(bbox_xyxy, 0, None)

            x1, y1, x2, y2 = bbox_xyxy
            left, right = sorted([x1, x2])
            top, bottom = sorted([y1, y2])

            if bottom <= top or right <= left:
                continue

            bbox_yxyx = [int(top), int(left), int(bottom), int(right)]
            predictions.append(PredictionBBox(class_name, bbox_yxyx, float(score)))
        return predictions

    def _format_predictions(
        self, labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, settings: dict
    ) -> List[List[PredictionBBox]]:
        thres = settings["confidence_threshold"]
        predictions = [self._format_prediction(*args, thres) for args in zip(labels, boxes, scores)]
        return predictions

    # -------------------------- #

    # Converters --------------- #
    def export_onnx(self, deploy_params: dict) -> str:
        model_files = deploy_params["model_files"]
        model_source = deploy_params["model_source"]
        model_info = deploy_params["model_info"]
        checkpoint_path = model_files["checkpoint"]
        config_path = self._get_config_path(model_source, model_info, model_files)
        checkpoint_path = export_onnx(checkpoint_path, config_path, self.model_dir)
        return checkpoint_path

    def export_tensorrt(self, deploy_params: dict) -> str:
        model_files = deploy_params["model_files"]
        model_source = deploy_params["model_source"]
        model_info = deploy_params["model_info"]
        checkpoint_path = model_files["checkpoint"]
        config_path = self._get_config_path(model_source, model_info, model_files)
        checkpoint_path = export_onnx(checkpoint_path, config_path, self.model_dir)
        checkpoint_path = export_tensorrt(checkpoint_path, self.model_dir, fp16=True)
        return checkpoint_path

    # -------------------------- #

    # Utils -------------------- #
    def _prepare_custom_model(self, model_files: dict):
        checkpoint_path = model_files["checkpoint"]
        config_path = model_files["config"]
        self._remove_include(config_path)
        return checkpoint_path, config_path

    def _prepare_pretrained_model(self, model_files: dict, model_info: dict):
        checkpoint_path = model_files["checkpoint"]
        model_name = model_info["meta"]["model_name"]
        if model_name.startswith("DEIM D-FINE"):
            CONFIG_DIR = "configs/deim_dfine"
        elif model_name.startswith("DEIM RT-DETRv2"):
            CONFIG_DIR = "configs/deim_rtdetrv2"
        else:
            CONFIG_DIR = "configs/deimv2"
        config_path = f'{CONFIG_DIR}/{get_file_name_with_ext(model_files["config"])}'
        self.classes = list(mscoco_category2name.values())
        obj_classes = [sly.ObjClass(name, sly.Rectangle) for name in self.classes]
        tag_metas = [sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)]
        self._model_meta = sly.ProjectMeta(obj_classes=obj_classes, tag_metas=tag_metas)
        self.checkpoint_info = CheckpointInfo(
            checkpoint_name=os.path.basename(checkpoint_path),
            model_name=model_info["meta"]["model_name"],
            architecture=self.FRAMEWORK_NAME,
            checkpoint_url=model_info["meta"]["model_files"]["checkpoint"],
            model_source=ModelSource.PRETRAINED,
        )
        return checkpoint_path, config_path

    def _get_config_path(self, model_source: str, model_info: dict, model_files: dict):
        if model_source == ModelSource.PRETRAINED:
            model_name = model_info["meta"]["model_name"]
            if model_name.startswith("DEIM D-FINE"):
                CONFIG_DIR = "configs/deim_dfine"
            elif model_name.startswith("DEIM RT-DETRv2"):
                CONFIG_DIR = "configs/deim_rtdetrv2"
            else:
                CONFIG_DIR = "configs/deimv2"
            config_path = f'{CONFIG_DIR}/{get_file_name_with_ext(model_files["config"])}'
        else:
            config_path = model_files["config"]
        return config_path

    def _remove_include(self, config_path: str):
        # del "__include__" and rewrite the config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "__include__" in config:
            config.pop("__include__")
            with open(config_path, "w") as f:
                yaml.dump(config, f)

    def _remove_existing_checkpoints(self, checkpoint_path: str, format: str):
        if format == "onnx":
            onnx_path = checkpoint_path.replace(".pth", ".onnx")
            if os.path.exists(onnx_path):
                sly.fs.silent_remove(onnx_path)
        elif format == "engine":
            onnx_path = checkpoint_path.replace(".pth", ".onnx")
            engine_path = checkpoint_path.replace(".pth", ".engine")
            if os.path.exists(onnx_path):
                sly.fs.silent_remove(onnx_path)
            if os.path.exists(engine_path):
                sly.fs.silent_remove(engine_path)

    def _clear_global_config(self):
        import importlib
        import sys
        from collections import defaultdict

        import engine.core.workspace

        for module_name in list(sys.modules.keys()):
            if module_name.startswith("engine."):
                importlib.reload(sys.modules[module_name])
        engine.core.workspace.GLOBAL_CONFIG = defaultdict(dict)

    # -------------------------- #
