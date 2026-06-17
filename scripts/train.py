import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataLoader import buildDataLoaders
from src.data.dataSplit import createSplits, SplitConfig
from src.models.modelFactory import buildModel, buildParamGroups, loadPretrainedWeights
from src.training.checkpoint import saveJson, loadCheckpoint
from src.training.trainer import Trainer
from src.utils.configUtils import loadTrainingConfig, TrainingConfig
from src.utils.colabUtils import setupColabOutput


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PlantDoc AI baseline model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint if it exists")
    parser.add_argument("--useGdrive", action="store_true", help="Mount Google Drive and save artifacts there if running in Colab")
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=None,
        help=(
            "Path to a local .pt checkpoint to use as pretrained base "
            "(overrides config.pretrainedPath and ImageNet pretrained). "
            "Supports both full PlantDocAI checkpoints and raw state_dicts."
        ),
    )
    return parser.parse_args()


def resolveDevice(deviceArg: str) -> torch.device:
    """Resolve device string to torch.device, with CUDA availability check."""
    if deviceArg == "cpu":
        return torch.device("cpu")
    if deviceArg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Fallback to CPU.")
        return torch.device("cpu")
    # "auto" or "cuda" (when available)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def countParameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in a model."""
    totalParams = sum(p.numel() for p in model.parameters())
    trainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return totalParams, trainableParams


def buildScheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    numEpochs: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Tạo learning rate scheduler dựa trên config.schedulerType.

    Modes:
      "step"          → StepLR(step_size=3, gamma=0.1) — behavior mặc định cũ.
      "cosine"        → CosineAnnealingLR(T_max=numEpochs).
      "warmup_cosine" → Linear warmup trong warmupEpochs epoch đầu,
                        sau đó CosineAnnealingLR cho các epoch còn lại.
                        Warmup: LR tăng đều từ 0 → learningRate.

    Args:
        optimizer: Optimizer đã được tạo.
        config: TrainingConfig chứa schedulerType, warmupEpochs.
        numEpochs: Tổng số epoch training.

    Returns:
        Scheduler object hoặc None nếu schedulerType không hợp lệ.
    """
    schedulerType = config.schedulerType.lower().strip()

    if schedulerType == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        print(f"[INFO] Scheduler: StepLR (step_size=3, gamma=0.1)")

    elif schedulerType == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)
        print(f"[INFO] Scheduler: CosineAnnealingLR (T_max={numEpochs})")

    elif schedulerType == "warmup_cosine":
        warmupEpochs = config.warmupEpochs
        cosineEpochs = max(1, numEpochs - warmupEpochs)

        if warmupEpochs <= 0:
            # Không có warmup → fallback về cosine thuần
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=numEpochs)
            print(f"[INFO] Scheduler: CosineAnnealingLR (warmupEpochs=0, T_max={numEpochs})")
        else:
            # LinearLR: LR tăng từ start_factor=1/warmupEpochs → 1.0 trong warmupEpochs epoch
            warmupScheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / warmupEpochs,
                end_factor=1.0,
                total_iters=warmupEpochs,
            )
            cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cosineEpochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmupScheduler, cosineScheduler],
                milestones=[warmupEpochs],
            )
            print(
                f"[INFO] Scheduler: warmup_cosine "
                f"(warmup={warmupEpochs} epochs -> cosine T_max={cosineEpochs} epochs)"
            )

    else:
        print(f"[WARN] schedulerType='{schedulerType}' không hợp lệ. Fallback về StepLR.")
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    return scheduler


def main() -> None:
    args = parseArgs()
    
    # Load typed config object, no more hidden fallbacks
    config = loadTrainingConfig(args.config)
    device = resolveDevice(config.device)

    # CLI --pretrained-path ghi đè config.pretrainedPath nếu được cung cấp
    if args.pretrained_path is not None:
        config.pretrainedPath = args.pretrained_path
        print(f"[INFO] --pretrained-path override: {config.pretrainedPath}")

    # Validate or prepare splits
    requiredSplits = ["train.csv", "val.csv", "test.csv", "classes.csv"]
    hasAllSplits = all((Path(config.splitDir) / f).exists() for f in requiredSplits)

    if not hasAllSplits:
        print(f"[INFO] Missing or incomplete split files in '{config.splitDir}'. Auto-generating splits...")
        createSplits(dataDir=config.dataDir, outDir=config.splitDir, splitConfig=SplitConfig())
        print("[INFO] Split generation completed.")
    else:
        print(f"[INFO] Found existing split files in '{config.splitDir}'.")

    # Build Dataloaders
    loaders, classToId = buildDataLoaders(
        dataDir=config.dataDir,
        splitDir=config.splitDir,
        inputSize=config.imageSize,
        batchSize=config.batchSize,
        numWorkers=config.numWorkers,
        aug=config.augConfig,
        useWeightedSampler=config.useWeightedSampler,
    )
    
    idToClass = {v: k for k, v in classToId.items()}
    classNames = [idToClass[i] for i in range(len(classToId))]
    numClasses = len(classNames)

    # ── Build Model ────────────────────────────────────────────────────────────
    # Khi useStagedFinetuning=True, giai đoạn 1 bắt buộc freeze backbone.
    # freezeBackbone trong config chỉ có hiệu lực khi useStagedFinetuning=False.
    effectiveFreezeBackbone = config.freezeBackbone
    if config.useStagedFinetuning:
        effectiveFreezeBackbone = True
        print(
            f"[INFO] useStagedFinetuning=True → override freezeBackbone=True cho giai đoạn 1 "
            f"(headOnlyEpochs={config.headOnlyEpochs})"
        )

    model = buildModel(
        modelName=config.modelName,
        numClasses=numClasses,
        usePretrained=(not bool(config.pretrainedPath)),
        freezeBackbone=effectiveFreezeBackbone,
    ).to(device)

    # ── Load custom pretrained weights nếu có ─────────────────────────────────
    if config.pretrainedPath:
        loadPretrainedWeights(model, config.pretrainedPath)
        # Áp dụng lại freeze sau khi load weights (loadPretrainedWeights không thay đổi requires_grad)
        if effectiveFreezeBackbone:
            from src.models.modelFactory import freezeModel, unfreezeModel
            freezeModel(model)
            classifier = model.get_classifier()
            if classifier is not None:
                unfreezeModel(classifier)
            print("[INFO] Freeze backbone reapplied sau khi load custom pretrained weights.")
    else:
        print(f"[INFO] Using ImageNet pretrained weights (timm).")

    totalParams, trainableParams = countParameters(model)
    print(f"[INFO] Model: {config.modelName} | freezeBackbone={effectiveFreezeBackbone} | staged={config.useStagedFinetuning}")
    print(f"[INFO] Params: {totalParams:,} (Total) | {trainableParams:,} (Trainable)")

    # ── Output paths ───────────────────────────────────────────────────────────
    outputDirStr = setupColabOutput(config.outputDir, args.useGdrive)
    outputDir = Path(outputDirStr)
    outputDir.mkdir(parents=True, exist_ok=True)


    # ── Handle Checkpoint Resume (PEEK) ────────────────────────────────────────
    # Chúng ta cần biết epoch của checkpoint TRƯỚC khi tạo optimizer
    # để đảm bảo số lượng parameter groups khớp nhau (staged unfreeze).
    startEpoch = 1
    checkpointData = None
    if args.resume:
        lastCheckpointPath = outputDir / "checkpoints" / "last.pt"
        if lastCheckpointPath.exists():
            print(f"[INFO] Peeking at checkpoint to resolve training stage: {lastCheckpointPath}")
            checkpointData = torch.load(str(lastCheckpointPath), map_location="cpu")
            checkpointEpoch = checkpointData.get("epoch", 0)
            startEpoch = checkpointEpoch + 1
            
            # Nếu checkpoint ĐÃ QUA giai đoạn unfreeze, chúng ta phải unfreeze model ngay bây giờ
            if config.useStagedFinetuning and checkpointEpoch > config.headOnlyEpochs:
                print(f"[INFO] Checkpoint epoch {checkpointEpoch} > headOnlyEpochs {config.headOnlyEpochs}.")
                print("[INFO] Unfreezing model EARLY to match checkpoint optimizer state.")
                from src.models.modelFactory import unfreezeModel
                unfreezeModel(model)
                effectiveFreezeBackbone = False
                
            # Kiểm tra xem checkpoint cũ có mấy optimizer parameter groups.
            # Nếu checkpoint cũ (trước bản update) chỉ có 1 group, mà config mới đòi 2 groups (useLayerLR),
            # thì pytorch sẽ báo lỗi ValueError. Do đó cần auto fallback.
            if "optimizerStateDict" in checkpointData:
                ckpt_groups = len(checkpointData["optimizerStateDict"]["param_groups"])
                if config.useLayerLR and not effectiveFreezeBackbone and ckpt_groups == 1:
                    print(f"⚠️ [WARNING] Checkpoint có {ckpt_groups} optimizer group, nhưng config yêu cầu Layer-wise LR (2 groups).")
                    print(f"⚠️ [WARNING] Tự động TẮT useLayerLR để có thể Resume thành công từ checkpoint cũ.")
                    config.useLayerLR = False
        else:
            print(f"[INFO] --resume flag given but {lastCheckpointPath} not found. Starting fresh.")

    # ── Loss ───────────────────────────────────────────────────────────────────
    if config.useClassWeights:
        from collections import Counter
        from src.data.dataSplit import loadSplitCsv
        
        trainSamples = loadSplitCsv(f"{config.splitDir}/train.csv")
        labelCounts = Counter(s.labelId for s in trainSamples)
        total = sum(labelCounts.values())
        
        # Calculate inverse frequency weights to balance classes
        weights = [total / (numClasses * labelCounts[i]) for i in range(numClasses)]
        classWeightsTensor = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=classWeightsTensor)
        print(f"[INFO] CrossEntropyLoss uses calculated class weights.")
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimizer ──────────────────────────────────────────────────────────────
    if config.useLayerLR and not effectiveFreezeBackbone:
        # Layer-wise LR chỉ có nghĩa khi backbone không bị freeze
        paramGroups = buildParamGroups(
            model=model,
            headLR=config.learningRate,
            backboneLR=config.backboneLR,
        )
    elif config.useLayerLR and effectiveFreezeBackbone:
        # Staged mode giai đoạn 1: chỉ head trainable → dùng 1 group
        print("[INFO] useLayerLR=True nhưng backbone đang bị freeze (giai đoạn 1). Dùng single param group.")
        paramGroups = [p for p in model.parameters() if p.requires_grad]
    else:
        # Behavior cũ: 1 param group duy nhất
        paramGroups = [p for p in model.parameters() if p.requires_grad]
        
    optimizer = torch.optim.Adam(
        paramGroups,
        lr=config.learningRate,
        weight_decay=config.weightDecay,
    )

    # ── Scheduler ──────────────────────────────────────────────────────────────
    scheduler = buildScheduler(optimizer=optimizer, config=config, numEpochs=config.numEpochs)

    # ── Load Full Checkpoint State ─────────────────────────────────────────────
    if checkpointData is not None:
        model.load_state_dict(checkpointData["modelStateDict"])
        if "optimizerStateDict" in checkpointData:
            optimizer.load_state_dict(checkpointData["optimizerStateDict"])
        if "schedulerStateDict" in checkpointData and scheduler is not None:
            scheduler.load_state_dict(checkpointData["schedulerStateDict"])
        print(f"[INFO] Restored model, optimizer, and scheduler state from epoch {startEpoch-1}.")

    # Prepare runtime config dictionary for saving
    runtimeConfig = config.to_dict()
    runtimeConfig["numClasses"] = numClasses
    runtimeConfig["classNames"] = classNames
    saveJson(str(outputDir / "config.json"), runtimeConfig)

    # ── Build Trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        device=device,
        trainLoader=loaders["train"],
        valLoader=loaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        classNames=classNames,
        outputDir=str(outputDir),
        topK=config.topK,
        saveBestMetric="valTop1",
        # Staged fine-tuning params
        useStagedFinetuning=config.useStagedFinetuning,
        headOnlyEpochs=config.headOnlyEpochs,
        # Optimizer rebuild config (Trainer cần để rebuild optimizer sau staged unfreeze)
        optimizerLR=config.learningRate,
        weightDecay=config.weightDecay,
        backboneLR=config.backboneLR if config.useLayerLR else None,
        useLayerLR=config.useLayerLR,
        stageUnfreezeApplied=(not effectiveFreezeBackbone),
    )

    if checkpointData is not None:
        trainer.bestMetric = checkpointData.get("bestMetric", float("-inf"))

    # ── Start Training ─────────────────────────────────────────────────────────
    trainer.fit(numEpochs=config.numEpochs, config=runtimeConfig, startEpoch=startEpoch)


if __name__ == "__main__":
    main()