import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.data.dataTransforms import AugConfig

@dataclass
class TrainingConfig:
    experimentName: str = "baseline"
    dataDir: str = "data/PlantVillage/train"
    splitDir: str = "data/splits"
    outputDir: str = "artifacts/baseline"
    modelName: str = "mobilenetV2"
    imageSize: int = 224
    freezeBackbone: bool = False
    batchSize: int = 32
    numEpochs: int = 5
    learningRate: float = 0.001
    weightDecay: float = 0.0001
    numWorkers: int = 2
    topK: int = 3
    device: str = "auto"

    # ── Class imbalance handling ──────────────────────────────────────────────
    # useWeightedSampler: Bật WeightedRandomSampler cho train DataLoader.
    # Oversample class thiểu số để mỗi epoch thấy phân bố đều hơn.
    useWeightedSampler: bool = True

    # useClassWeights: Bật class weights trong CrossEntropyLoss.
    # Tăng loss weight cho class thiểu số.
    # CẢNH BÁO: Dùng cả sampler + class weights có nguy cơ overcompensation.
    # Khuyến nghị: bắt đầu với sampler only, chỉ bật nếu kết quả chưa đủ.
    useClassWeights: bool = False

    # ── Staged Fine-tuning ────────────────────────────────────────────────────
    # useStagedFinetuning: Bật chế độ staged — giai đoạn 1 chỉ train head,
    # giai đoạn 2 unfreeze toàn backbone và train với LR thấp hơn.
    # Khi bật, freezeBackbone sẽ tự động được set True cho giai đoạn 1.
    # Default: false → behavior hiện tại không thay đổi.
    useStagedFinetuning: bool = False

    # headOnlyEpochs: Số epoch chỉ train classifier head (giai đoạn 1).
    # Sau epoch này, backbone sẽ được unfreeze và optimizer được tạo lại.
    # Chỉ có hiệu lực khi useStagedFinetuning=true.
    headOnlyEpochs: int = 3

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # schedulerType: Loại learning rate scheduler.
    #   "step"          → StepLR(step_size=3, gamma=0.1) — behavior cũ
    #   "cosine"        → CosineAnnealingLR(T_max=numEpochs)
    #   "warmup_cosine" → Linear warmup rồi cosine decay
    schedulerType: str = "step"

    # warmupEpochs: Số epoch warmup LR từ 0 lên learningRate.
    # Chỉ có hiệu lực khi schedulerType="warmup_cosine".
    # Warmup giúp tránh shock gradient khi bắt đầu fine-tune backbone.
    warmupEpochs: int = 3

    # ── Layer-wise Learning Rate ──────────────────────────────────────────────
    # useLayerLR: Bật layer-wise LR — head và backbone dùng LR khác nhau.
    # Head: dùng learningRate (config chính)
    # Backbone: dùng backboneLR (thấp hơn để giữ ổn định pretrained features)
    # Default: false → behavior hiện tại (1 LR cho toàn model)
    useLayerLR: bool = False

    # backboneLR: LR cho backbone khi useLayerLR=true.
    # Thường đặt thấp hơn learningRate 5–10 lần để tránh catastrophic forgetting.
    backboneLR: float = 1e-4

    # ── Custom Pretrained Checkpoint ──────────────────────────────────────────
    # pretrainedPath: Đường dẫn tới file checkpoint .pt để load làm pretrained base.
    # Nếu rỗng ("") hoặc không đặt → dùng ImageNet pretrained weights từ timm.
    # Hỗ trợ cả format full checkpoint (có key "modelStateDict") và raw state_dict.
    # Dùng strict=False để cho phép classifier head khác numClasses.
    pretrainedPath: str = ""

    augConfig: AugConfig = field(default_factory=AugConfig)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered_dict = {k: v for k, v in d.items() if k in valid_keys and k != "augConfig"}

        # Parse nested `augmentation:` block from YAML into AugConfig
        aug_raw = d.get("augmentation", {})
        if isinstance(aug_raw, dict):
            aug_fields = {f for f in AugConfig.__dataclass_fields__}
            aug_filtered = {k: v for k, v in aug_raw.items() if k in aug_fields}
            filtered_dict["augConfig"] = AugConfig(**aug_filtered)
        else:
            filtered_dict["augConfig"] = AugConfig()

        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "augConfig"}
        # Serialise AugConfig as a nested dict under 'augmentation'
        result["augmentation"] = {k: getattr(self.augConfig, k) for k in self.augConfig.__dataclass_fields__}
        return result

def loadYamlConfig(configPath: str) -> dict:
    """Load a YAML configuration file as raw dictionary."""
    path = Path(configPath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {configPath}")
        
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        
    if config is None:
        config = {}
        
    return config

def loadTrainingConfig(configPath: str) -> TrainingConfig:
    """Load a YAML configuration file specifically into TrainingConfig."""
    raw_dict = loadYamlConfig(configPath)
    return TrainingConfig.from_dict(raw_dict)
