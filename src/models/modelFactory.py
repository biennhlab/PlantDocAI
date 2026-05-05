import torch
import torch.nn as nn
import timm
from pathlib import Path
from typing import Optional


def freezeModel(model):
    """Utility to freeze all parameters in a generic torch module."""
    for param in model.parameters():
        param.requires_grad = False

def unfreezeModel(model):
    """Utility to unfreeze all parameters in a generic torch module."""
    for param in model.parameters():
        param.requires_grad = True

def buildModel(modelName: str, numClasses: int, usePretrained: bool = True, freezeBackbone: bool = True):
    """
    Builds a PyTorch image classification model using `timm`.
    
    Args:
        modelName (str): Name of the architecture (e.g., 'mobilenetv2_100', 'resnet50', 'convnext_tiny').
        numClasses (int): Number of target classes for the final classification layer.
        usePretrained (bool): If True, loads pre-trained ImageNet weights via timm.
                              Ignored if pretrainedPath is used after buildModel().
        freezeBackbone (bool): If True, freezes all layers except the newly initialized classifier head.
        
    Returns:
        torch.nn.Module: The configured PyTorch model ready for training.
    """
    try:
        # timm automatically handles replacing the final classifier head to match num_classes
        model = timm.create_model(modelName, pretrained=usePretrained, num_classes=numClasses)
    except Exception as e:
        raise ValueError(f"Failed to create model '{modelName}' using timm. Is the model name valid in timm? Error: {e}")

    if freezeBackbone:
        # Freeze the entire model originally
        freezeModel(model)
        
        # Then safely unfreeze only the classifier head
        classifier = model.get_classifier()
        if classifier is not None:
            unfreezeModel(classifier)
            
    return model


def buildParamGroups(model: nn.Module, headLR: float, backboneLR: float) -> list:
    """
    Tách model thành 2 param groups với learning rate khác nhau:
      - Classifier head: headLR (cao hơn — head được khởi tạo ngẫu nhiên)
      - Backbone layers: backboneLR (thấp hơn — bảo toàn pretrained features)

    Dùng id() để phân biệt params thuộc head vs backbone, tránh hardcode tên layer.

    Args:
        model: PyTorch model (timm-based).
        headLR: Learning rate cho classifier head.
        backboneLR: Learning rate cho backbone layers.

    Returns:
        List of param group dicts phù hợp với torch.optim.
    """
    classifier = model.get_classifier()
    if classifier is None:
        # Không tách được head → trả về 1 group duy nhất
        print("[WARN] buildParamGroups: get_classifier() returned None. Falling back to single param group.")
        trainableParams = [p for p in model.parameters() if p.requires_grad]
        print(f"[INFO] Param groups: single group — {len(trainableParams)} params @ lr={headLR}")
        return [{"params": trainableParams, "lr": headLR}]

    # Lấy id của tất cả params trong head
    headParamIds = {id(p) for p in classifier.parameters()}

    headParams = [p for p in model.parameters() if id(p) in headParamIds and p.requires_grad]
    backboneParams = [p for p in model.parameters() if id(p) not in headParamIds and p.requires_grad]

    print(
        f"[INFO] Param groups: "
        f"head={len(headParams)} params @ lr={headLR:.2e} | "
        f"backbone={len(backboneParams)} params @ lr={backboneLR:.2e}"
    )

    paramGroups = []
    if headParams:
        paramGroups.append({"params": headParams, "lr": headLR})
    if backboneParams:
        paramGroups.append({"params": backboneParams, "lr": backboneLR})

    if not paramGroups:
        raise RuntimeError(
            "buildParamGroups: No trainable parameters found. "
            "Check freezeBackbone config or model.parameters()."
        )

    return paramGroups


def loadPretrainedWeights(model: nn.Module, pretrainedPath: str) -> None:
    """
    Load trọng số pretrained từ file checkpoint cục bộ vào model.

    Hỗ trợ 2 format checkpoint:
      1. Full training checkpoint: dict có key "modelStateDict" (format PlantDocAI)
      2. Raw state_dict: dict ánh xạ trực tiếp param_name → tensor

    Dùng strict=False để cho phép mismatch ở classifier head (numClasses khác).
    Keys bị bỏ qua hoặc thiếu sẽ được log rõ ràng.

    Args:
        model: PyTorch model đã được khởi tạo (với classifier head phù hợp numClasses mới).
        pretrainedPath: Đường dẫn tới file .pt checkpoint.
    """
    path = Path(pretrainedPath)
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] pretrainedPath không tồn tại: {pretrainedPath}")

    print(f"[INFO] Loading pretrained weights from: {pretrainedPath}")
    raw = torch.load(str(path), map_location="cpu")

    # Phân biệt full checkpoint vs raw state_dict
    if isinstance(raw, dict) and "modelStateDict" in raw:
        stateDict = raw["modelStateDict"]
        srcEpoch = raw.get("epoch", "?")
        srcMetric = raw.get("bestMetric", "?")
        srcClasses = len(raw.get("classNames", []))
        print(
            f"[INFO] Full checkpoint detected — source epoch={srcEpoch}, "
            f"bestMetric={srcMetric:.4f}, numClasses={srcClasses}"
            if isinstance(srcMetric, float)
            else f"[INFO] Full checkpoint detected — source epoch={srcEpoch}, numClasses={srcClasses}"
        )
    else:
        stateDict = raw
        print("[INFO] Raw state_dict detected.")

    result = model.load_state_dict(stateDict, strict=False)

    if result.missing_keys:
        print(f"[INFO] Keys missing in checkpoint (will use random init): {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[INFO] Keys in checkpoint not in model (skipped): {result.unexpected_keys}")

    if not result.missing_keys and not result.unexpected_keys:
        print("[INFO] Pretrained weights loaded successfully — perfect match.")
    else:
        print("[INFO] Pretrained weights loaded with partial match (expected if numClasses changed).")