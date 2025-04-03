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

    def load_model(
        self, model_files: dict, model_info: dict, model_source: str, device: str, runtime: str
    ):
        checkpoint_path = model_files["checkpoint"]
        if model_source == ModelSource.CUSTOM:
            config_path = model_files["config"]
            self._remove_include(config_path)
        else:
            model_name = model_info["meta"]["model_name"]
            if model_name.startswith("DEIM D-FINE"):
                CONFIG_DIR = "configs/deim_dfine"
            else:
                CONFIG_DIR = "configs/deim_rtdetrv2"
            config_path = f'{CONFIG_DIR}/{get_file_name_with_ext(model_files["config"])}'
            self.classes = list(mscoco_category2name.values())
            self.checkpoint_info = CheckpointInfo(
                checkpoint_name=os.path.basename(checkpoint_path),
                model_name=model_info["meta"]["model_name"],
                architecture=self.FRAMEWORK_NAME,
                checkpoint_url=model_info["meta"]["model_files"]["checkpoint"],
                model_source=model_source,
            )

        h, w = 640, 640
        self.img_size = [w, h]
        self.transforms = T.Compose(
            [
                T.Resize((h, w)),
                T.ToTensor(),
            ]
        )

        if runtime == RuntimeType.PYTORCH:
            self._load_pytorch(checkpoint_path, config_path, device)
        elif runtime == RuntimeType.ONNXRUNTIME:
            self._load_runtime(checkpoint_path, config_path, "onnx", device)
        elif runtime == RuntimeType.TENSORRT:
            self._load_runtime(checkpoint_path, config_path, "tensorrt", device)

    def _load_runtime(self, checkpoint_path: str, config_path: str, format: str, device: str):
        def export_model():
            sly.logger.info(f"Exporting model to '{format}' format...")
            if self.gui is not None:
                bar = self.gui.download_progress(
                    message=f"Exporting model to '{format}' format...", total=1
                )
                self.gui.download_progress.show()
            if format == "onnx":
                onnx_model_path = export_onnx(checkpoint_path, config_path, self.model_dir)
            elif format == "tensorrt":
                onnx_model_path = export_onnx(checkpoint_path, config_path, self.model_dir)
                engine_path = export_tensorrt(onnx_model_path, self.model_dir, fp16=True)
            if self.gui is not None:
                bar.update(1)
                self.gui.download_progress.hide()
            if format == "onnx":
                return onnx_model_path
            elif format == "tensorrt":
                return engine_path

        file_name = sly.fs.get_file_name(checkpoint_path)
        file_ext = sly.fs.get_file_ext(checkpoint_path)
        if file_name == "best" and file_ext == ".pth":
            exported_checkpoint_path = export_model()
        elif file_ext == ".pth":
            exported_checkpoint_path = checkpoint_path.replace(".pth", f".{format}")
            if not os.path.exists(exported_checkpoint_path):
                exported_checkpoint_path = export_model()
        else:
            exported_checkpoint_path = checkpoint_path   

        if format == "onnx":
            self._load_onnx(exported_checkpoint_path, device)
        elif format == "tensorrt":
            self._load_tensorrt(exported_checkpoint_path, device)

    def _load_pytorch(self, checkpoint_path: str, config_path: str, device: str):
        self.cfg = YAMLConfig(config_path, resume=checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        self.model = self.cfg.model
        self.model.load_state_dict(state)
        self.model.deploy().to(device)
        self.postprocessor = self.cfg.postprocessor.deploy().to(device)

    def _load_onnx(self, onnx_path: str, device: str):
        import onnxruntime
        providers = (
            ["CUDAExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
        )
        if device != "cpu":
            assert (
                onnxruntime.get_device() == "GPU"
            ), "ONNXRuntime is not configured to use GPU"
        self.onnx_session = onnxruntime.InferenceSession(
            onnx_path, providers=providers
        )

    def _load_tensorrt(self, engine_path: str, device: str):
        from tools.inference.trt_inf import TRTInference
        assert device != "cpu", "TensorRT is not supported on CPU"
        self.engine = TRTInference(engine_path, device)
        self.max_batch_size = 1

    def predict_benchmark(self, images_np: List[np.ndarray], settings: dict = None):
        if self.runtime == RuntimeType.PYTORCH:
            return self._predict_pytorch(images_np, settings)
        elif self.runtime == RuntimeType.ONNXRUNTIME:
            return self._predict_onnx(images_np, settings)
        elif self.runtime == RuntimeType.TENSORRT:
            return self._predict_tensorrt(images_np, settings)

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
        size_input = torch.tensor([self.img_size * len(images_np)]).reshape(-1, 2)
        return img_input.to(device), size_input.to(device), orig_sizes.to(device)

    def _format_prediction(
        self, labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, conf_tresh: float
    ) -> List[PredictionBBox]:
        predictions = []
        for label, bbox_xyxy, score in zip(labels, boxes, scores):
            if score < conf_tresh:
                continue
            class_name = self.classes[label]
            bbox_xyxy = np.round(bbox_xyxy).astype(int)
            bbox_xyxy = np.clip(bbox_xyxy, 0, None)
            bbox_yxyx = [bbox_xyxy[1], bbox_xyxy[0], bbox_xyxy[3], bbox_xyxy[2]]
            bbox_yxyx = list(map(int, bbox_yxyx))
            predictions.append(PredictionBBox(class_name, bbox_yxyx, float(score)))
        return predictions

    def _format_predictions(
        self, labels: np.ndarray, boxes: np.ndarray, scores: np.ndarray, settings: dict
    ) -> List[List[PredictionBBox]]:
        thres = settings["confidence_threshold"]
        predictions = [self._format_prediction(*args, thres) for args in zip(labels, boxes, scores)]
        return predictions

    def _remove_include(self, config_path: str):
        # del "__include__" and rewrite the config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if "__include__" in config:
            config.pop("__include__")
            with open(config_path, "w") as f:
                yaml.dump(config, f)
