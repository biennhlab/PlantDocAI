# src/evaluation/predictor.py
"""
Inference Pipeline — PlantDocAI.

Module này đóng vai trò cầu nối giữa model đã train và ứng dụng thực tế.
Tách biệt hoàn toàn logic suy luận khỏi notebook/script, cung cấp API ổn định
để Streamlit app hoặc bất kỳ consumer nào có thể gọi.

Thiết kế:
  - Tái sử dụng buildModel() từ modelFactory
  - Tái sử dụng loadCheckpoint() từ checkpoint module
  - Tái sử dụng buildInferenceTransform() từ dataTransforms (single source of truth)
  - Class names đọc từ config.json artifact (được tạo khi train)
  - Output chuẩn hóa: classId, className, confidence (post-softmax)
  - Tích hợp Energy-based OOD Detection: tính energy score từ logits thô
  - Tích hợp Grad-CAM tùy chọn: sinh heatmap overlay qua explain() / explainFromPil()
"""

import json
import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

from src.models.modelFactory import buildModel
from src.training.checkpoint import loadCheckpoint, findCheckpoint
from src.data.dataTransforms import buildInferenceTransform
from src.validation.ood_detector import (
    EnergyOODDetector,
    OODResult,
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Pipeline suy luận tái sử dụng cho PlantDocAI.

    Đóng gói toàn bộ: load config → build model → load weights → preprocess → predict.
    App chỉ cần khởi tạo một lần, rồi gọi predict() hoặc predictFromPil() nhiều lần.

    Output format::

        {
            "predictions": [
                {"classId": int, "className": str, "confidence": float},
                ...
            ],
            "ood": {
                "is_ood": bool,
                "energy_score": float,
                "threshold": float,
                "temperature": float,
                "warning_message": str | None
            }
        }

    Usage::

        pipeline = InferencePipeline(modelDir="artifacts/mobilenetV2_colab_artifacts")

        # Predict từ file path
        result = pipeline.predict("path/to/leaf.jpg", topK=3)
        predictions = result["predictions"]
        ood_info = result["ood"]

        # Predict từ PIL Image (Streamlit)
        result = pipeline.predictFromPil(pil_image, topK=3)

        # Top-1 tiện lợi
        result = pipeline.predictTop1FromPil(pil_image)
    """

    def __init__(self, modelDir: Union[str, Path], device: str = "auto") -> None:
        """
        Khởi tạo InferencePipeline.

        Args:
            modelDir: Thư mục chứa ``config.json`` và checkpoint
                      (e.g. ``artifacts/mobilenetV2_colab_artifacts``).
            device: Device chạy inference ('cpu', 'cuda', hoặc 'auto').
        """
        self.modelDir = Path(modelDir)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._loadConfig()
        self._initModel()
        self._initTransforms()
        self._initOODDetector()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization helpers (private)
    # ─────────────────────────────────────────────────────────────────────────

    def _loadConfig(self) -> None:
        """Load config.json artifact được tạo khi train."""
        configPath = self.modelDir / "config.json"
        if not configPath.exists():
            raise FileNotFoundError(f"Cannot find config at {configPath}")

        with open(configPath, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Parse class names — có thể là list hoặc string representation của list
        classNames = self.config.get("classNames", [])
        if isinstance(classNames, str):
            classNames = ast.literal_eval(classNames)
        self.classNames = classNames
        self.numClasses = self.config.get("numClasses", len(self.classNames))

        # Model params
        self.modelName = self.config.get("modelName", "mobilenetv2_100")
        self.imageSize = self.config.get("imageSize", 224)

    def _initModel(self) -> None:
        """Build model via timm factory và load checkpoint weights."""
        # Tìm checkpoint theo thứ tự ưu tiên
        weightsPath = findCheckpoint(self.modelDir)

        if weightsPath is None:
            raise FileNotFoundError(
                f"Cannot find checkpoint (e.g., best.pt) in {self.modelDir}"
            )

        # Build model qua modelFactory — usePretrained=False vì sẽ load weights từ checkpoint
        self.model = buildModel(
            modelName=self.modelName,
            numClasses=self.numClasses,
            usePretrained=False,
            freezeBackbone=False,
        )

        # Load weights từ checkpoint (dict format: { modelStateDict, ... })
        loadCheckpoint(
            checkpointPath=str(weightsPath),
            model=self.model,
            mapLocation=str(self.device),
        )

        self.model.to(self.device)
        self.model.eval()

    def _initTransforms(self) -> None:
        """
        Build preprocessing transform cho inference.

        Dùng buildInferenceTransform() từ dataTransforms.py — đây là single source of truth,
        đảm bảo preprocessing khớp chính xác với eval pipeline dùng khi train:
        Resize(inputSize × 1.14) → CenterCrop(inputSize) → ToTensor → Normalize(ImageNet).
        """
        self.transform = buildInferenceTransform(self.imageSize)

    def _initOODDetector(self) -> None:
        """
        Khởi tạo Energy-based OOD Detector.

        Đọc cấu hình OOD từ config.json nếu có (key 'ood'),
        fallback về default constants trong ood_detector.py.
        """
        oodConfig = self.config.get("ood", {})
        threshold = oodConfig.get("energyThreshold", DEFAULT_ENERGY_THRESHOLD)
        temperature = oodConfig.get("temperature", DEFAULT_TEMPERATURE)

        self.oodDetector = EnergyOODDetector(
            threshold=threshold,
            temperature=temperature,
        )
        logger.info(
            "OOD Detector initialized: threshold=%.2f, temperature=%.2f",
            threshold, temperature,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public prediction API
    # ─────────────────────────────────────────────────────────────────────────

    def predictFromPil(
        self, image: Image.Image, topK: int = 3
    ) -> Dict[str, Union[List, dict]]:
        """
        Predict trên một PIL Image.

        Đây là method cốt lõi — tất cả các predict methods khác đều delegate về đây.
        Phù hợp nhất cho Streamlit (nhận ảnh upload trực tiếp dưới dạng PIL).

        Luồng xử lý:
        1. Preprocess ảnh (Resize → CenterCrop → Normalize)
        2. Forward model → lấy logits thô
        3. Tính Energy Score từ logits → OOD detection
        4. Softmax → Top-K predictions

        Args:
            image: PIL Image (sẽ được convert sang RGB nếu cần).
            topK: Số lượng dự đoán top-k trả về (mặc định 3).

        Returns:
            Dict gồm:
              - ``predictions``: List[Dict] top-k predictions, sắp xếp giảm dần
                theo confidence. Mỗi item: {classId, className, confidence}.
              - ``ood``: Dict kết quả OOD detection
                {is_ood, energy_score, threshold, temperature, warning_message}.
        """
        # Đảm bảo RGB — ảnh có thể là RGBA, L, P,...
        image = image.convert("RGB")

        # Preprocess: Resize → CenterCrop → ToTensor → Normalize
        tensorImg = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensorImg)

            # ── OOD Detection từ logits thô ──────────────────────────────
            oodResult = self.oodDetector.detect(logits)

            # ── Softmax → Top-K predictions ──────────────────────────────
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

        # Top-k results, đã sắp xếp giảm dần theo confidence
        actualK = min(topK, self.numClasses)
        topProb, topClassIdx = torch.topk(probabilities, k=actualK)

        predictions = []
        for i in range(actualK):
            idx = topClassIdx[i].item()
            prob = topProb[i].item()
            name = self.classNames[idx] if idx < len(self.classNames) else f"Class_{idx}"
            predictions.append({
                "classId": idx,
                "className": name,
                "confidence": prob,
            })

        return {
            "predictions": predictions,
            "ood": oodResult.to_dict(),
        }

    def predict(
        self, imagePath: Union[str, Path], topK: int = 3
    ) -> Dict[str, Union[List, dict]]:
        """
        Predict trên một ảnh từ đường dẫn file.

        Args:
            imagePath: Đường dẫn tới file ảnh.
            topK: Số lượng dự đoán top-k trả về.

        Returns:
            Dict: Xem ``predictFromPil()``.
        """
        try:
            with Image.open(imagePath) as img:
                # Copy ra ngoài context manager để tránh file handle issue
                img = img.copy()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image for prediction at {imagePath}. Error: {e}"
            )

        return self.predictFromPil(img, topK=topK)

    def predictTop1FromPil(
        self, image: Image.Image
    ) -> Dict[str, Union[str, int, float, dict]]:
        """
        Predict top-1 trên PIL Image. Trả về dict gồm prediction + OOD.

        Args:
            image: PIL Image.

        Returns:
            Dict: ``{classId, className, confidence, ood: {...}}``
        """
        result = self.predictFromPil(image, topK=1)
        top1 = result["predictions"][0]
        top1["ood"] = result["ood"]
        return top1

    def predictTop1(
        self, imagePath: Union[str, Path]
    ) -> Dict[str, Union[str, int, float, dict]]:
        """
        Predict top-1 từ đường dẫn file. Trả về dict gồm prediction + OOD.

        Args:
            imagePath: Đường dẫn tới file ảnh.

        Returns:
            Dict: ``{classId, className, confidence, ood: {...}}``
        """
        try:
            with Image.open(imagePath) as img:
                img = img.copy()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image for prediction at {imagePath}. Error: {e}"
            )

        return self.predictTop1FromPil(img)

    # ─────────────────────────────────────────────────────────────────────────
    # Explainability API (Grad-CAM)
    # ─────────────────────────────────────────────────────────────────────────

    def explainFromPil(
        self,
        image: Image.Image,
        topK: int = 3,
        class_idx: Optional[int] = None,
        alpha: float = 0.5,
    ) -> Dict[str, Union[List, Image.Image, int, dict]]:
        """
        Predict + sinh Grad-CAM overlay trên PIL Image.

        Quy trình:
        1. Preprocess ảnh bằng cùng transform với predict (single source of truth).
        2. Chạy Grad-CAM trên preprocessed tensor → heatmap.
        3. Overlay heatmap lên ảnh gốc → PIL.Image sẵn sàng cho Streamlit.
        4. Chạy predict bình thường để lấy top-k results + OOD.

        Args:
            image: PIL Image gốc (chưa preprocess).
            topK: Số lượng top-k predictions.
            class_idx: Index class cần giải thích. Nếu None, dùng predicted class (argmax).
            alpha: Độ đậm của heatmap overlay (0.0–1.0).

        Returns:
            Dict gồm:
              - ``predictions``: List[Dict] top-k predictions (giống predictFromPil).
              - ``ood``: Dict kết quả OOD detection.
              - ``gradcamOverlay``: PIL.Image overlay heatmap lên ảnh gốc.
              - ``targetClassIdx``: int — class index được dùng để sinh heatmap.

        Note:
            Grad-CAM hooks được tạo và hủy mỗi lần gọi để tránh memory leak
            khi chạy nhiều lần trong Streamlit app.
        """
        from src.explain.gradcam import GradCAM

        # Đảm bảo RGB
        image = image.convert("RGB")

        # Preprocess bằng cùng transform với inference — single source of truth
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Sinh Grad-CAM (hooks tạo + hủy tự động qua context manager)
        with GradCAM(self.model, model_name=self.modelName, device=self.device) as cam:
            heatmap = cam.generate(input_tensor, class_idx=class_idx)
            target_idx = class_idx if class_idx is not None else input_tensor.new_tensor(0).long().item()

            # Xác định target class index thực tế (argmax nếu chưa chỉ định)
            if class_idx is None:
                with torch.no_grad():
                    logits = self.model(input_tensor)
                    target_idx = logits.argmax(dim=1).item()
            else:
                target_idx = class_idx

            overlay = cam.overlay_on_image(image, heatmap, alpha=alpha)

        # Model trở lại eval mode ổn định — predict bình thường (bao gồm OOD)
        self.model.eval()
        result = self.predictFromPil(image, topK=topK)

        return {
            "predictions": result["predictions"],
            "ood": result["ood"],
            "gradcamOverlay": overlay,
            "targetClassIdx": target_idx,
        }

    def explain(
        self,
        imagePath: Union[str, Path],
        topK: int = 3,
        class_idx: Optional[int] = None,
        alpha: float = 0.5,
    ) -> Dict[str, Union[List, Image.Image, int, dict]]:
        """
        Predict + sinh Grad-CAM overlay từ đường dẫn file ảnh.

        Args:
            imagePath: Đường dẫn tới file ảnh.
            topK: Số lượng top-k predictions.
            class_idx: Index class cần giải thích (None = predicted class).
            alpha: Độ đậm overlay.

        Returns:
            Dict: Xem ``explainFromPil()``.
        """
        try:
            with Image.open(imagePath) as img:
                img = img.copy()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image for explanation at {imagePath}. Error: {e}"
            )

        return self.explainFromPil(img, topK=topK, class_idx=class_idx, alpha=alpha)
