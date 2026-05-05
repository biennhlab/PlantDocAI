# src/data/dataTransforms.py
"""
Pipeline Augmentation có nhận thức về bệnh lá cây — PlantVillage / PlantDocAI.

Giải thích từng quyết định thiết kế:
──────────────────────────────────────────────────────────────────
1. Hue bị KHÓA ở 0.0  — màu sắc bệnh (vàng lá, nâu đốm, đỏ tổn thương) là tín hiệu
   chẩn đoán quan trọng.  Thay đổi hue sẽ tổng hợp ra trạng thái bệnh giả hoặc
   lật ngược hoàn toàn tín hiệu bệnh học.

2. Saturation được phép thay đổi nhẹ — ảnh thực địa có saturation biến thiên do
   camera / ánh sáng khác nhau, nhưng giữ rất bảo thủ để tránh làm phai màu bệnh.

3. Brightness & Contrast ở mức trung bình — biến động ánh sáng là thực tế (chụp trong
   nhà/ngoài trời), nhưng được cap để không làm mờ hoặc tối quá texture của tổn thương.

4. Rotation đủ 360° — lá được chụp từ mọi hướng ngẫu nhiên. Xoay toàn bộ góc là
   hợp lệ về mặt sinh học và giúp chống bias góc nhìn rất hiệu quả.

5. GaussianBlur tùy chọn & nhẹ — mô phỏng rung tay / ống kính ướt khi chụp ảnh thực địa.
   Áp dụng với xác suất thấp để không làm nhòa texture vi mô của tổn thương.

6. RandomErasing bị tránh dùng — có thể che khuất tổn thương duy nhất trên một chiếc lá
   nhỏ, xóa đi tín hiệu chẩn đoán duy nhất.  Thay vào đó dùng GridDistortion.

7. Elastic/Grid Distortion nhẹ — mô phỏng độ cong / nhăn của bề mặt lá.  Biên độ thấp
   để không phá hủy hình dạng tổn thương (vòng tròn của early blight, target spot).

8. CutMix / MixUp KHÔNG có ở đây — trộn hai mẫu bệnh tạo ra nhãn mơ hồ.  Với các lớp
   đã rất giống nhau (Tomato Early Blight vs Target Spot), điều này làm tệ hơn sự nhầm lẫn.
   Triển khai riêng chỉ sau khi phân tích pairwise classifier xác nhận tính tách biệt.

9. RandomErasing vùng nhỏ — xác suất thấp xóa patch nhỏ với giá trị fill trung tính (grey)
   giúp tránh model ghi nhớ texture nền cụ thể mà không xóa tổn thương trên vùng foreground
   (patch rất nhỏ: 2–10% diện tích ảnh).

10. CoarseDropout (nhiều patch nhỏ) — phá texture nền đồng nhất hiệu quả hơn 1 patch
    RandomErasing. Nhiều lỗ nhỏ phân bố đều buộc model không ghi nhớ vùng nền cụ thể.

11. GaussNoise — thêm nhiễu Gaussian vào pixel, phá vỡ sự đồng nhất pixel-level của
    nền sạch (xám / trắng), buộc model tập trung vào structural features thay vì pixel values.

12. RandomGrayscale — xác suất rất thấp (5%) chuyển ảnh sang grayscale.  Buộc model học
    cả shape/texture, không chỉ dựa vào màu nền.  Giữ thấp vì màu bệnh là tín hiệu quan trọng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torchvision import transforms

# ── Thử import tùy chọn (albumentations cho elastic / grid) ─────────────────
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import numpy as np
    _HAS_ALBUMENTATIONS = True
except ImportError:
    _HAS_ALBUMENTATIONS = False

# ─────────────────────────────────────────────────────────────────────────────
# Thống số chuẩn hóa ImageNet (dùng cho mọi model được pretrain trên ImageNet,
# bao gồm MobileNetV2, EfficientNet, ResNet variants)
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


@dataclass
class AugConfig:
    """
    Các tham số augmentation phẳng, có thể serialize sang YAML.
    Tất cả giá trị đều có default an toàn và bảo thủ.
    Ghi đè qua configs/baseline.yaml → augmentation: {...}
    """

    # ── Hình họbc ──────────────────────────────────────────────────────────────
    # Xoay đầy đủ 180° vì hướng lá ngẫu nhiên trong ảnh thực địa.
    rotationDegrees: int = 180

    # Scale của RandomResizedCrop: tối thiểu 0.7 đảm bảo đủ ngữ cảnh quanh tổn thương.
    # KHÔNG giảm dưới 0.60 — nguy cơ crop mất tổn thương trên lá nhỏ.
    cropScaleMin: float = 0.70
    cropScaleMax: float = 1.00

    # Flip ngang + dọc: lá bị lật vẫn là cùng một bệnh.
    hflip: bool = True
    vflip: bool = True

    # Biến dạng đàn hồi nhẹ: mô phỏng độ cong / nhăn bề mặt lá.
    # Chỉ áp dụng nếu albumentations được cài đặt.
    elasticEnabled: bool = True
    elasticAlpha: float = 30.0   # biên độ dịch chuyển — giữ thấp
    elasticSigma: float = 5.0    # hệ số làm mượt — giữ >= 4 để tránh artifact pixel
    elasticP: float = 0.30       # áp dụng 30% thời gian

    # ── Màu sắc (hue bị KHÓA ở 0.0) ─────────────────────────────────────────
    # Độ sáng: biến động ánh sáng thực địa tự nhiên — phạm vi an toàn.
    brightnessJitter: float = 0.25

    # Độ tương phản: biến động do mây che / bóng râm.
    contrastJitter: float = 0.20

    # Saturation: dịch chuyển white-balance camera — giữ thấp.
    saturationJitter: float = 0.10

    # Hue: BẮT BUỘC giữ ở 0.0.  Màu bệnh (vàng, đốm nâu, đốm đỏ)
    # là đặc trưng chẩn đoán chính.  Thay đổi hue tổng hợp ra mẫu giả y học.
    hueJitter: float = 0.00

    # ── Làm mờ ───────────────────────────────────────────────────────────────
    # Mô phỏng rung tay / ẩm ướt trên ống kính khi chụp ngoài thực địa.
    blurEnabled: bool = True
    blurKernelMin: int = 3       # kernel tối thiểu (phải lẻ)
    blurKernelMax: int = 5       # giữ nhỏ — bảo toàn cạnh texture tổn thương
    blurSigmaMin: float = 0.1
    blurSigmaMax: float = 1.0
    blurP: float = 0.20          # xác suất 20% — không làm mờ hầu hết mẫu

    # ── Xóa vùng nhỏ (chống bias nền, KHÔNG xóa tổn thương) ─────────────────
    # Xóa một vùng cực nhỏ (2–8% ảnh) với fill màu xám trung tính.
    # Mục đích: ngăn model ghi nhớ patch texture nền cụ thể.
    # Scale được giữ rất nhỏ — đây KHÔNG phải RandomErasing thông thường cho
    # regularization (patch lớn có nguy cơ che tổn thương).
    eraseEnabled: bool = True
    eraseScaleMin: float = 0.02
    eraseScaleMax: float = 0.08
    eraseRatio: float = 1.0      # chỉ patch vuông
    eraseP: float = 0.20

    # ── Độ sắc nét ───────────────────────────────────────────────────────────
    # Đôi khi làm sắc nét để mô phỏng output của camera thực địa độ phân giải cao.
    sharpnessEnabled: bool = True
    sharpnessFactor: float = 1.5   # >1 = sắc hơn; 1.0 = không thay đổi
    sharpnessP: float = 0.20

    # ── CoarseDropout (nhiều patch nhỏ — chống shortcut nền) ─────────────────
    # Xóa nhiều patch nhỏ phân bố đều trên ảnh thay vì 1 patch lớn.
    # Hiệu quả hơn RandomErasing đơn lẻ trong việc phá texture nền đồng nhất.
    # Kích thước patch nhỏ (16×16 trên ảnh 224×224 ≈ 0.5% diện tích/hole)
    # đảm bảo không che khuất toàn bộ tổn thương.
    coarseDropoutEnabled: bool = True
    coarseDropoutMaxHoles: int = 6
    coarseDropoutHoleHeight: int = 16
    coarseDropoutHoleWidth: int = 16
    coarseDropoutP: float = 0.30

    # ── Gaussian Noise (phá nền sạch) ────────────────────────────────────────
    # Thêm nhiễu Gaussian vào pixel — phá vỡ sự đồng nhất pixel-level của
    # nền sạch (xám / trắng) trong PlantVillage.
    # varLimit thấp (5–25) để không phá hủy texture vi mô của tổn thương.
    gaussNoiseEnabled: bool = True
    gaussNoiseVarMin: float = 5.0
    gaussNoiseVarMax: float = 25.0
    gaussNoiseP: float = 0.25

    # ── Random Grayscale (buộc học shape thay vì chỉ dựa màu nền) ────────────
    # Xác suất rất thấp (5%) để không mất tín hiệu màu bệnh quan trọng.
    # Mục đích: buộc model học cả contour/shape, không chỉ dựa vào color channel.
    grayscaleP: float = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Hàm hỗ trợ công khai — được gọi bởi buildTransforms()
# ─────────────────────────────────────────────────────────────────────────────

def _buildTorchvisionTrain(inputSize: int, aug: AugConfig) -> transforms.Compose:
    """
    Transform training chỉ dùng torchvision.  Được dùng khi albumentations
    chưa được cài đặt, hoặc khi elasticEnabled=False.
    """
    ops: list = [
        transforms.RandomResizedCrop(
            inputSize,
            scale=(aug.cropScaleMin, aug.cropScaleMax),
            ratio=(0.85, 1.18),   # biến động tỉ lệ khung hình vừa phải
        ),
    ]

    if aug.hflip:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if aug.vflip:
        ops.append(transforms.RandomVerticalFlip(p=0.5))

    ops.append(transforms.RandomRotation(degrees=aug.rotationDegrees))

    # hueJitter được truyền tường minh — mặc định là 0.0.
    # Nếu người dùng đặt hueJitter != 0, sẽ phát cảnh báo runtime.
    if aug.hueJitter != 0.0:
        import warnings
        warnings.warn(
            f"[AugConfig] hueJitter={aug.hueJitter} khác 0. "
            "Màu sắc bệnh lá là tín hiệu chẩn đoán quan trọng — "
            "thay đổi hue có thể tổng hợp ra mẫu không hợp lệ về y học. "
            "Đặt hueJitter=0.0 trừ khi có lý do cụ thể từ domain.",
            UserWarning,
            stacklevel=2,
        )

    ops.append(
        transforms.ColorJitter(
            brightness=aug.brightnessJitter,
            contrast=aug.contrastJitter,
            saturation=aug.saturationJitter,
            hue=aug.hueJitter,       # 0.0 — bị khóa
        )
    )

    if aug.blurEnabled:
        # GaussianBlur của torchvision yêu cầu kernel_size là số lẻ.
        kernel_size = aug.blurKernelMax if aug.blurKernelMax % 2 == 1 else aug.blurKernelMax + 1
        ops.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(
                    kernel_size=kernel_size,
                    sigma=(aug.blurSigmaMin, aug.blurSigmaMax),
                )],
                p=aug.blurP,
            )
        )

    if aug.sharpnessEnabled:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomAdjustSharpness(sharpness_factor=aug.sharpnessFactor)],
                p=aug.sharpnessP,
            )
        )

    # ── RandomGrayscale (buộc model học shape, không chỉ dựa color) ───────
    if aug.grayscaleP > 0:
        ops.append(transforms.RandomGrayscale(p=aug.grayscaleP))

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    # Xóa vùng nền nhỏ: áp dụng SAU Normalize để giá trị fill 0
    # tương đương ≈ xám (pixel chuẩn hóa theo ImageNet mean) — trung tính, không phải đen thuần.
    if aug.eraseEnabled:
        ops.append(
            transforms.RandomErasing(
                p=aug.eraseP,
                scale=(aug.eraseScaleMin, aug.eraseScaleMax),
                ratio=(aug.eraseRatio, aug.eraseRatio),
                value=0,         # 0 sau normalize ≈ ImageNet mean (trung tính)
                inplace=False,
            )
        )

    # ── Gaussian Noise fallback (torchvision) ─────────────────────────────
    # Torchvision không có GaussNoise native → custom transform trên tensor.
    if aug.gaussNoiseEnabled:
        ops.append(_TorchGaussianNoise(
            var_min=aug.gaussNoiseVarMin,
            var_max=aug.gaussNoiseVarMax,
            p=aug.gaussNoiseP,
        ))

    # ── CoarseDropout fallback (torchvision) ──────────────────────────────
    # Torchvision không có CoarseDropout → custom transform nhiều lỗ nhỏ trên tensor.
    if aug.coarseDropoutEnabled:
        ops.append(_TorchCoarseDropout(
            max_holes=aug.coarseDropoutMaxHoles,
            hole_height=aug.coarseDropoutHoleHeight,
            hole_width=aug.coarseDropoutHoleWidth,
            p=aug.coarseDropoutP,
        ))

    return transforms.Compose(ops)


def _buildAlbumentationsTrain(inputSize: int, aug: AugConfig):
    """
    Pipeline augmentation với albumentations cho elastic/grid distortion.
    Trả về wrapper callable để interface giống torchvision transforms.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import numpy as np

    albu_ops = [
        A.RandomResizedCrop(
            size=(inputSize, inputSize),
            scale=(aug.cropScaleMin, aug.cropScaleMax),
            ratio=(0.85, 1.18),
        ),
        A.HorizontalFlip(p=0.5 if aug.hflip else 0.0),
        A.VerticalFlip(p=0.5 if aug.vflip else 0.0),
        A.Rotate(limit=aug.rotationDegrees, p=1.0),
        A.ColorJitter(
            brightness=aug.brightnessJitter,
            contrast=aug.contrastJitter,
            saturation=aug.saturationJitter,
            hue=aug.hueJitter,    # 0.0 — bị khóa
            p=0.8,
        ),
    ]

    if aug.blurEnabled:
        albu_ops.append(
            A.GaussianBlur(
                blur_limit=(aug.blurKernelMin, aug.blurKernelMax),
                sigma_limit=(aug.blurSigmaMin, aug.blurSigmaMax),
                p=aug.blurP,
            )
        )

    if aug.sharpnessEnabled:
        albu_ops.append(A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.0), p=aug.sharpnessP))

    if aug.elasticEnabled:
        # ElasticTransform: biến dạng nhẹ mô phỏng độ cong bề mặt lá.
        # alpha=30, sigma=5 → uốn cong nhẹ, không làm méo mạnh.
        albu_ops.append(
            A.ElasticTransform(
                alpha=aug.elasticAlpha,
                sigma=aug.elasticSigma,
                p=aug.elasticP,
            )
        )

    # ── GaussNoise (phá nền sạch — trước normalize) ──────────────────────
    if aug.gaussNoiseEnabled:
        # Albumentations 1.4+ dùng std_range, phiên bản cũ dùng var_limit.
        # Chúng ta ưu tiên API mới để xóa warning trên Colab.
        try:
            # Ước lượng std từ variance (sqrt(var))
            std_min = aug.gaussNoiseVarMin ** 0.5
            std_max = aug.gaussNoiseVarMax ** 0.5
            albu_ops.append(A.GaussNoise(std_range=(std_min, std_max), p=aug.gaussNoiseP))
        except (TypeError, ValueError):
            albu_ops.append(A.GaussNoise(var_limit=(aug.gaussNoiseVarMin, aug.gaussNoiseVarMax), p=aug.gaussNoiseP))

    # ── CoarseDropout (nhiều patch nhỏ — chống shortcut nền) ─────────────
    if aug.coarseDropoutEnabled:
        try:
            # Albumentations 1.4+ API
            albu_ops.append(
                A.CoarseDropout(
                    num_holes_range=(1, aug.coarseDropoutMaxHoles),
                    hole_height_range=(8, aug.coarseDropoutHoleHeight),
                    hole_width_range=(8, aug.coarseDropoutHoleWidth),
                    fill_value=128,
                    p=aug.coarseDropoutP,
                )
            )
        except (TypeError, ValueError):
            albu_ops.append(
                A.CoarseDropout(
                    max_holes=aug.coarseDropoutMaxHoles,
                    max_height=aug.coarseDropoutHoleHeight,
                    max_width=aug.coarseDropoutHoleWidth,
                    fill_value=128,
                    p=aug.coarseDropoutP,
                )
            )

    # ── RandomGrayscale (buộc model học shape) ───────────────────────────
    if aug.grayscaleP > 0:
        albu_ops.append(A.ToGray(p=aug.grayscaleP))

    albu_ops += [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    pipeline = A.Compose(albu_ops)

    class _AlbuWrapper:
        """Bọc pipeline albumentations để nhận PIL.Image (giống torchvision)."""

        def __init__(self, pipeline):
            self._p = pipeline

        def __call__(self, pil_image):
            img_np = np.array(pil_image)   # H x W x 3, uint8
            result = self._p(image=img_np)["image"]  # tensor sau ToTensorV2
            # Áp dụng xóa vùng nhỏ SAU normalize (trên tensor)
            if aug.eraseEnabled:
                result = transforms.RandomErasing(
                    p=aug.eraseP,
                    scale=(aug.eraseScaleMin, aug.eraseScaleMax),
                    ratio=(aug.eraseRatio, aug.eraseRatio),
                    value=0,
                    inplace=False,
                )(result)
            return result

    return _AlbuWrapper(pipeline)


# ─────────────────────────────────────────────────────────────────────────────
# Custom torchvision-only transforms (fallback khi không có albumentations)
# ─────────────────────────────────────────────────────────────────────────────

class _TorchGaussianNoise:
    """
    Thêm nhiễu Gaussian vào tensor (SAU normalize).
    Fallback cho albumentations GaussNoise khi chỉ dùng torchvision.
    """

    def __init__(self, var_min: float = 5.0, var_max: float = 25.0, p: float = 0.25):
        # var_limit từ albumentations là variance trên ảnh uint8 (0-255).
        # Sau normalize ImageNet, cần scale tương ứng: std = sqrt(var) / 255.
        self.std_min = (var_min ** 0.5) / 255.0
        self.std_max = (var_max ** 0.5) / 255.0
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return tensor
        std = self.std_min + torch.rand(1).item() * (self.std_max - self.std_min)
        noise = torch.randn_like(tensor) * std
        return tensor + noise


class _TorchCoarseDropout:
    """
    Xóa nhiều patch nhỏ ngẫu nhiên trên tensor (SAU normalize).
    Fallback cho albumentations CoarseDropout khi chỉ dùng torchvision.
    Fill value = 0 (sau normalize theo ImageNet mean = xám trung tính).
    """

    def __init__(self, max_holes: int = 6, hole_height: int = 16,
                 hole_width: int = 16, p: float = 0.30):
        self.max_holes = max_holes
        self.hole_height = hole_height
        self.hole_width = hole_width
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return tensor
        _, h, w = tensor.shape
        n_holes = torch.randint(1, self.max_holes + 1, (1,)).item()
        tensor = tensor.clone()
        for _ in range(n_holes):
            y = torch.randint(0, max(1, h - self.hole_height), (1,)).item()
            x = torch.randint(0, max(1, w - self.hole_width), (1,)).item()
            y2 = min(y + self.hole_height, h)
            x2 = min(x + self.hole_width, w)
            tensor[:, y:y2, x:x2] = 0  # 0 sau normalize = xám trung tính
        return tensor


def buildInferenceTransform(inputSize: int = 224) -> transforms.Compose:
    """
    Build transform chuẩn cho inference / evaluation.
    Khớp chính xác với eval pipeline dùng khi train (Resize 1.14× → CenterCrop).

    Đây là single source of truth — cả buildTransforms() lẫn InferencePipeline
    đều gọi hàm này để đảm bảo preprocessing nhất quán.

    Args:
        inputSize: Kích thước vuông đầu vào cho model (mặc định 224).

    Returns:
        transforms.Compose: Pipeline transform cho inference.
    """
    return transforms.Compose([
        transforms.Resize(int(inputSize * 1.14)),
        transforms.CenterCrop(inputSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def buildTransforms(
    inputSize: int = 224,
    aug: Optional[AugConfig] = None,
) -> Dict[str, object]:
    """
    Xây dựng dictionary transform cho train / val / test.

    Args:
        inputSize: Kích thước vuông đầu vào cho model.
        aug:       Đối tượng AugConfig.  Nếu None, dùng default bảo thủ.

    Returns:
        dict với keys "train", "val", "test".
    """
    if aug is None:
        aug = AugConfig()

    # ── Transform training ────────────────────────────────────────────────────
    if _HAS_ALBUMENTATIONS and aug.elasticEnabled:
        trainTf = _buildAlbumentationsTrain(inputSize, aug)
    else:
        trainTf = _buildTorchvisionTrain(inputSize, aug)

    # ── Transform eval (val + test) ───────────────────────────────────────────
    # Dùng chung buildInferenceTransform() để đảm bảo nhất quán với inference.
    evalTf = buildInferenceTransform(inputSize)

    return {"train": trainTf, "val": evalTf, "test": evalTf}