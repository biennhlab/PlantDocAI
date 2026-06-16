import argparse
import os
from pathlib import Path

import torch
from torch import nn
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataLoader import buildDataLoaders
from src.data.dataSplit import createSplits, SplitConfig
from src.models.modelFactory import buildModel
from src.training.checkpoint import saveJson, loadCheckpoint
from src.training.trainer import Trainer
from src.utils.configUtils import loadTrainingConfig
from src.utils.colabUtils import setupColabOutput

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PlantDoc AI baseline model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint if it exists")
    parser.add_argument("--useGdrive", action="store_true", help="Mount Google Drive and save artifacts there if running in Colab")
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

def main() -> None:
    args = parseArgs()
    
    # Load typed config object, no more hidden fallbacks
    config = loadTrainingConfig(args.config)
    device = resolveDevice(config.device)

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

    # Build Model
    model = buildModel(
        modelName=config.modelName,
        numClasses=numClasses,
        usePretrained=True,
        freezeBackbone=config.freezeBackbone,
    ).to(device)

    totalParams, trainableParams = countParameters(model)
    print(f"[INFO] Model: {config.modelName} | Freeze Backbone: {config.freezeBackbone}")
    print(f"[INFO] Params: {totalParams:,} (Total) | {trainableParams:,} (Trainable)")

    # Loss and Optimizer
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
        
    trainableModelParams = [param for param in model.parameters() if param.requires_grad]
    
    optimizer = torch.optim.Adam(
        trainableModelParams,
        lr=config.learningRate,
        weight_decay=config.weightDecay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Prepare logic isolation for output paths (Google Drive aware)
    outputDirStr = setupColabOutput(config.outputDir, args.useGdrive)
    outputDir = Path(outputDirStr)
    outputDir.mkdir(parents=True, exist_ok=True)

    # Prepare runtime config dictionary for saving
    runtimeConfig = config.to_dict()
    runtimeConfig["numClasses"] = numClasses
    runtimeConfig["classNames"] = classNames
    saveJson(str(outputDir / "config.json"), runtimeConfig)

    # Build Trainer
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
    )

    # Handle Checkpoint Resume
    startEpoch = 1
    if args.resume:
        lastCheckpointPath = outputDir / "checkpoints" / "last.pt"
        if lastCheckpointPath.exists():
            print(f"[INFO] Resuming from checkpoint: {lastCheckpointPath}")
            checkpoint = loadCheckpoint(
                checkpointPath=str(lastCheckpointPath),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                mapLocation=str(device)
            )
            startEpoch = checkpoint.get("epoch", 0) + 1
            trainer.bestMetric = checkpoint.get("bestMetric", float("-inf"))
            print(f"[INFO] Restored state. Resuming from epoch {startEpoch}...")
        else:
            print(f"[INFO] --resume flag given but {lastCheckpointPath} not found. Starting fresh.")

    # Start Training
    trainer.fit(numEpochs=config.numEpochs, config=runtimeConfig, startEpoch=startEpoch)

if __name__ == "__main__":
    main()