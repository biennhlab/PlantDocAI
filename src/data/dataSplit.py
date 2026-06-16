# src/data/dataSplit.py
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

from .plantVillageDataset import PlantVillageDataset, SampleItem

@dataclass
class SplitConfig:
    trainRatio: float = 0.8
    valRatio: float = 0.1
    testRatio: float = 0.1
    seed: int = 42

def createSplits(
    dataDir: str,
    outDir: str,
    splitConfig: SplitConfig,
) -> Dict[str, str]:
    """Create stratified train/val/test splits and save as CSV files."""
    os.makedirs(outDir, exist_ok=True)

    baseDs = PlantVillageDataset(rootDir=dataDir, samples=None, transform=None)
    samples = baseDs.samples
    labels = [s.labelId for s in samples]

    if abs(splitConfig.trainRatio + splitConfig.valRatio + splitConfig.testRatio - 1.0) > 1e-6:
        raise ValueError("trainRatio + valRatio + testRatio must sum to 1.0")

    tempRatio = splitConfig.valRatio + splitConfig.testRatio
    trainSamples, tempSamples, trainLabels, tempLabels = train_test_split(
        samples,
        labels,
        test_size=tempRatio,
        random_state=splitConfig.seed,
        stratify=labels,
    )

    if splitConfig.testRatio == 0:
        valSamples, testSamples = tempSamples, []
    else:
        testPortionOfTemp = splitConfig.testRatio / tempRatio
        valSamples, testSamples, _, _ = train_test_split(
            tempSamples,
            tempLabels,
            test_size=testPortionOfTemp,
            random_state=splitConfig.seed,
            stratify=tempLabels,
        )

    paths = {
        "train": os.path.join(outDir, "train.csv"),
        "val": os.path.join(outDir, "val.csv"),
        "test": os.path.join(outDir, "test.csv"),
        "classes": os.path.join(outDir, "classes.csv"),
    }

    _writeSamplesCsv(paths["train"], trainSamples)
    _writeSamplesCsv(paths["val"], valSamples)
    _writeSamplesCsv(paths["test"], testSamples)

    classToId, _ = baseDs.getClassMapping()
    _writeClassesCsv(paths["classes"], classToId)

    return paths

def loadSplitCsv(csvPath: str) -> List[SampleItem]:
    """Load a split CSV file into a list of SampleItem."""
    if not os.path.isfile(csvPath):
        raise FileNotFoundError(f"Split CSV not found: {csvPath}")

    samples: List[SampleItem] = []
    with open(csvPath, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            imagePath = row["imagePath"].strip()
            labelId = int(row["labelId"])
            samples.append(SampleItem(imagePath=imagePath, labelId=labelId))
    if len(samples) == 0:
        raise ValueError(f"No rows loaded from split CSV: {csvPath}")
    return samples

def _writeSamplesCsv(outPath: str, samples: List[SampleItem]) -> None:
    with open(outPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["imagePath", "labelId"])
        writer.writeheader()
        for s in samples:
            writer.writerow({"imagePath": s.imagePath, "labelId": s.labelId})

def _writeClassesCsv(outPath: str, classToId: Dict[str, int]) -> None:
    with open(outPath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["className", "labelId"])
        writer.writeheader()
        for className, labelId in sorted(classToId.items(), key=lambda x: x[1]):
            writer.writerow({"className": className, "labelId": labelId})