import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataLoader import buildDataLoaders
from src.models.modelFactory import buildModel
from src.training.checkpoint import loadCheckpoint, findCheckpoint

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def parseArgs():
    parser = argparse.ArgumentParser(description="Evaluate a trained PlantDoc AI model")
    parser.add_argument("--modelDir", type=str, required=True, help="Path to the model artifact directory (e.g., artifacts/baseline)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Which dataset split to evaluate on")
    parser.add_argument("--batchSize", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--saveMisclassified", action="store_true", help="Save misclassified images for debugging")
    return parser.parse_args()

def main():
    args = parseArgs()
    modelDir = Path(args.modelDir)
    
    if not modelDir.exists():
        logging.error(f"Model directory {modelDir} does not exist.")
        return
        
    configPath = modelDir / "config.json"
    if not configPath.exists():
        logging.error(f"Config file not found: {configPath}")
        return
        
    with open(configPath, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 1. Build DataLoaders (requesting paths explicitly for misclassified tracking)
    logging.info(f"Building DataLoader for '{args.split}' split...")
    loaders, classToId = buildDataLoaders(
        dataDir=config.get("dataDir", "data/PlantVillage/train"),
        splitDir=config.get("splitDir", "data/splits"),
        inputSize=config.get("imageSize", 224),
        batchSize=args.batchSize,
        numWorkers=config.get("numWorkers", 2),
        returnPath=True  # Important: enable returning path for debugging
    )
    
    if args.split not in loaders:
        logging.error(f"Split '{args.split}' not found in dataloaders.")
        return
        
    loader = loaders[args.split]
    classNames = config.get("classNames")
    if not classNames:
        idToClass = {v: k for k, v in classToId.items()}
        classNames = [idToClass[i] for i in range(len(classToId))]
    numClasses = len(classNames)
    
    # 2. Build Model and load Checkpoint
    modelName = config.get("modelName", "mobilenetv2_100")
    # Legacy path: old checkpoints trained with torchvision mobilenet_v2
    # have different weight keys than timm's mobilenetv2_100.
    if modelName == "mobilenetV2":
        import torchvision.models as models
        logging.info(f"Building legacy torchvision model 'mobilenet_v2' with {numClasses} classes...")
        model = models.mobilenet_v2(pretrained=False, num_classes=numClasses).to(device)
    else:
        logging.info(f"Building model '{modelName}' with {numClasses} classes using timm...")
        model = buildModel(
            modelName=modelName,
            numClasses=numClasses,
            usePretrained=False,
            freezeBackbone=False
        ).to(device)
    
    weightsPath = findCheckpoint(modelDir)
    if not weightsPath:
        logging.error(f"Could not find checkpoint in {modelDir}")
        return
        
    logging.info(f"Loading checkpoint from {weightsPath}")
    loadCheckpoint(str(weightsPath), model=model, mapLocation=str(device))
    model.eval()
    
    # 3. Evaluation Loop
    allTargets = []
    allPreds = []
    misclassified = []
    
    logging.info(f"Starting evaluation over {len(loader)} batches...")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if batch is None:
                continue
                
            # If returnPath=True is successfully enforced, batch has 3 items
            if len(batch) == 3:
                images, targets, paths = batch
            else:
                images, targets = batch
                paths = [None] * len(images)
                
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Store for metrics
            targets_cpu = targets.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            
            allTargets.extend(targets_cpu)
            allPreds.extend(preds_cpu)
            
            # Track misclassified if requested
            if args.saveMisclassified:
                for i, (t, p) in enumerate(zip(targets_cpu, preds_cpu)):
                    if t != p and paths[i] is not None:
                        misclassified.append({
                            "path": paths[i],
                            "true_class": classNames[t],
                            "pred_class": classNames[p]
                        })
            
    # 4. Compute Metrics
    logging.info("\n=== Classification Report ===")
    report = classification_report(allTargets, allPreds, target_names=classNames, digits=4)
    print(report)
    
    reportPath = modelDir / f"classification_report_{args.split}.txt"
    with open(reportPath, "w", encoding="utf-8") as f:
        f.write(report)
    logging.info(f"Saved classification report to {reportPath}")
    
    # 5. Confusion Matrix
    logging.info("Generating Confusion Matrix...")
    cm = confusion_matrix(allTargets, allPreds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classNames, yticklabels=classNames)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {args.split.capitalize()}')
    plt.tight_layout()
    
    cmPath = modelDir / f"confusion_matrix_{args.split}.png"
    plt.savefig(cmPath, dpi=300)
    plt.close()
    logging.info(f"Saved confusion matrix to {cmPath}")
    
    # 6. Save misclassified images
    if args.saveMisclassified and misclassified:
        errDir = modelDir / f"misclassified_{args.split}"
        if errDir.exists():
            shutil.rmtree(errDir)
        errDir.mkdir(parents=True)
        
        logging.info(f"Saving {len(misclassified)} misclassified samples to {errDir}...")
        rootDir = config.get("dataDir", "data/PlantVillage/train")
        
        for item in tqdm(misclassified, desc="Copying images"):
            relPath = item["path"]
            # Construct absolute path if it is relative
            if not os.path.isabs(relPath):
                fullPath = Path(rootDir) / relPath
            else:
                fullPath = Path(relPath)
                
            folderName = f"True_{item['true_class']}__Pred_{item['pred_class']}"
            targetFolder = errDir / folderName
            targetFolder.mkdir(parents=True, exist_ok=True)
            
            try:
                shutil.copy2(fullPath, targetFolder / fullPath.name)
            except Exception as e:
                logging.warning(f"Could not copy {fullPath}: {e}")
                
        logging.info("Done copying misclassified images.")

if __name__ == "__main__":
    main()
