# src/validation/ood_detector.py
"""
Energy-based Out-of-Distribution (OOD) Detection — PlantDocAI.

Module phát hiện ảnh nằm ngoài phân phối dữ liệu huấn luyện (ảnh không phải lá cây,
ảnh đồ vật, ảnh động vật, v.v.) bằng phương pháp Energy Score trên logits.

Nguyên lý:
    Thay vì dùng Softmax confidence (dễ bị overconfident), module tính Energy Score
    trực tiếp từ logits thô của model. Ảnh In-Distribution (ID) thường có energy thấp
    (âm sâu), ảnh OOD thường có energy cao (gần 0 hoặc dương).

    E(x; T) = -T * log(sum(exp(f_i(x) / T)))

    Nếu E(x) > threshold → cảnh báo ảnh có thể không phù hợp.

References:
    Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.

Usage::

    from src.validation.ood_detector import EnergyOODDetector

    detector = EnergyOODDetector(threshold=-5.0, temperature=1.0)
    result = detector.detect(logits)

    if result.is_ood:
        print(result.warning_message)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Constants / Defaults
# ─────────────────────────────────────────────────────────────────────────────

# Threshold mặc định — giá trị khởi tạo ban đầu.
# ⚠️ CẦN TUNE LẠI bằng bộ validation gồm ~100 ảnh leaf (ID) + ~100 ảnh non-leaf (OOD).
# Xem scripts/tune_ood_threshold.py để biết cách tune.
DEFAULT_ENERGY_THRESHOLD: float = -5.0

# Temperature mặc định cho Energy Score. T=1 là chuẩn trong paper gốc.
DEFAULT_TEMPERATURE: float = 1.0

# Warning message mặc định khi phát hiện OOD
_OOD_WARNING_VI: str = (
    "Ảnh tải lên có thể không phải là lá cây hoặc nằm ngoài phạm vi dữ liệu "
    "hệ thống đã học. Kết quả dự đoán (nếu có) chỉ nên dùng để tham khảo."
)


# ─────────────────────────────────────────────────────────────────────────────
# OOD Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OODResult:
    """Kết quả phát hiện Out-of-Distribution.

    Attributes:
        is_ood: True nếu ảnh được đánh giá là OOD (energy > threshold).
        energy_score: Giá trị Energy Score tính từ logits.
        threshold: Ngưỡng phân tách ID/OOD đang sử dụng.
        temperature: Nhiệt độ T trong công thức Energy.
        warning_message: Thông điệp cảnh báo (None nếu ảnh hợp lệ).
    """

    is_ood: bool
    energy_score: float
    threshold: float
    temperature: float
    warning_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Chuyển OODResult thành dict thuần — tiện cho JSON/Streamlit."""
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Core Function
# ─────────────────────────────────────────────────────────────────────────────

def compute_energy_score(
    logits: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> float:
    """
    Tính Energy Score từ logits thô (chưa qua Softmax).

    Công thức: E(x; T) = -T * log(sum(exp(logits / T)))
    Sử dụng torch.logsumexp để đảm bảo numerical stability (tránh overflow/underflow).

    Args:
        logits: Tensor logits từ model, shape (1, C) hoặc (C,).
                C = số lượng classes.
        temperature: Nhiệt độ T > 0. Mặc định 1.0.

    Returns:
        float: Energy score. Giá trị âm sâu = ID, giá trị cao (gần 0) = OOD.

    Raises:
        ValueError: Nếu logits rỗng hoặc temperature <= 0.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature phải > 0, nhận được: {temperature}")

    if logits.numel() == 0:
        raise ValueError("Logits tensor rỗng, không thể tính Energy Score.")

    # Đảm bảo 2D: (batch, num_classes)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # E(x; T) = -T * logsumexp(logits / T, dim=1)
    # logsumexp tự xử lý numerical stability
    energy = -temperature * torch.logsumexp(logits / temperature, dim=1)

    # Trả về scalar float (batch size 1)
    return energy[0].item()


# ─────────────────────────────────────────────────────────────────────────────
# Detector Class
# ─────────────────────────────────────────────────────────────────────────────

class EnergyOODDetector:
    """
    Detector phát hiện Out-of-Distribution dựa trên Energy Score.

    Detector này là stateless — chỉ lưu trữ threshold và temperature.
    Có thể tái sử dụng cho nhiều ảnh mà không cần reset.

    Args:
        threshold: Ngưỡng phân tách. Nếu energy > threshold → OOD.
                   Mặc định: -5.0 (cần tune bằng validation set).
        temperature: Nhiệt độ cho Energy Score. Mặc định: 1.0.

    Usage::

        detector = EnergyOODDetector(threshold=-5.0, temperature=1.0)
        result = detector.detect(logits)
        print(result.is_ood, result.energy_score)
    """

    def __init__(
        self,
        threshold: float = DEFAULT_ENERGY_THRESHOLD,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        if temperature <= 0:
            raise ValueError(f"Temperature phải > 0, nhận được: {temperature}")

        self.threshold = threshold
        self.temperature = temperature

    def detect(self, logits: torch.Tensor) -> OODResult:
        """
        Phát hiện OOD từ logits.

        Args:
            logits: Tensor logits thô từ model, shape (1, C) hoặc (C,).

        Returns:
            OODResult: Kết quả gồm is_ood, energy_score, threshold, temperature,
                       và warning_message (nếu OOD).
        """
        energy_score = compute_energy_score(logits, self.temperature)
        is_ood = energy_score > self.threshold

        return OODResult(
            is_ood=is_ood,
            energy_score=energy_score,
            threshold=self.threshold,
            temperature=self.temperature,
            warning_message=_OOD_WARNING_VI if is_ood else None,
        )

    def __repr__(self) -> str:
        return (
            f"EnergyOODDetector(threshold={self.threshold}, "
            f"temperature={self.temperature})"
        )
