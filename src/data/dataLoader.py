# src/data/dataLoader.py
"""DataLoader construction with optional weighted sampling for class imbalance."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Optional, Tuple, Any, List

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .plantVillageDataset import PlantVillageDataset
from .dataSplit import loadSplitCsv
from .dataTransforms import buildTransforms, AugConfig

def buildDataLoaders(

    dataDir: str,
    splitDir: str,
    inputSize: int = 224,
    batchSize: int = 32,
    numWorkers: int = 2,
    pinMemory: Optional[bool] = None,
    returnPath: bool = False,
    aug: Optional[AugConfig] = None,
    useWeightedSampler: bool = False,
):
    """
    Xây dựng DataLoader cho train / val / test.

    Args:
        dataDir: Đường dẫn tới thư mục chứa ảnh theo class folders.
        splitDir: Đường dẫn tới thư mục chứa split CSV files.
        inputSize: Kích thước ảnh đầu vào cho model.
        batchSize: Kích thước batch.
        numWorkers: Số worker cho DataLoader.
        pinMemory: Pin memory cho CUDA (auto-detect nếu None).
        returnPath: Trả thêm image path nếu True.
        aug: Cấu hình augmentation cho train transforms.
        useWeightedSampler: Dùng WeightedRandomSampler cho train DataLoader
                           để oversample class thiểu số. Khi True, thay thế
                           shuffle=True bằng sampler.
    """

    tf = buildTransforms(inputSize=inputSize, aug=aug)

    trainSamples = loadSplitCsv(f"{splitDir}/train.csv")
    valSamples = loadSplitCsv(f"{splitDir}/val.csv")
    testSamples = loadSplitCsv(f"{splitDir}/test.csv")

    trainDs = PlantVillageDataset(rootDir=dataDir, samples=trainSamples, transform=tf["train"], returnPath=returnPath)
    valDs = PlantVillageDataset(rootDir=dataDir, samples=valSamples, transform=tf["val"], returnPath=returnPath)
    testDs = PlantVillageDataset(rootDir=dataDir, samples=testSamples, transform=tf["test"], returnPath=returnPath)

    if pinMemory is None:
        pinMemory = bool(torch.cuda.is_available())

    # ── Shared DataLoader kwargs ──────────────────────────────────────────
    commonKwargs = dict(
        batch_size=batchSize, num_workers=numWorkers,
        pin_memory=pinMemory, drop_last=False,
    )

    # ── Train DataLoader: sampler hoặc shuffle ────────────────────────────
    if useWeightedSampler:
        sampler = _buildWeightedSampler(trainSamples)
        print("[INFO] WeightedRandomSampler enabled — oversampling minority classes.")
        trainLoader = DataLoader(trainDs, sampler=sampler, **commonKwargs)
    else:
        trainLoader = DataLoader(trainDs, shuffle=True, **commonKwargs)

    valLoader = DataLoader(valDs, shuffle=False, **commonKwargs)
    testLoader = DataLoader(testDs, shuffle=False, **commonKwargs)

    scanDs = PlantVillageDataset(rootDir=dataDir, samples=None, transform=None)
    classToId, _ = scanDs.getClassMapping()

    return {"train": trainLoader, "val": valLoader, "test": testLoader}, classToId


def _buildWeightedSampler(samples) -> WeightedRandomSampler:
    """
    Tạo WeightedRandomSampler từ danh sách SampleItem.

    Trọng số mỗi sample = N / count(class_of_sample), nghĩa là class có ít
    mẫu hơn sẽ được sample nhiều hơn, đảm bảo mỗi epoch model thấy phân bố
    gần đều giữa các class.

    Args:
        samples: Danh sách SampleItem (imagePath, labelId).

    Returns:
        WeightedRandomSampler với replacement=True và num_samples = len(samples).
    """
    labelCounts = Counter(s.labelId for s in samples)
    numSamples = len(samples)

    # Trọng số class: inverse frequency — class càng ít thì weight càng cao.
    classWeights = {cid: numSamples / cnt for cid, cnt in labelCounts.items()}

    # Trọng số từng sample theo class của nó.
    sampleWeights = [classWeights[s.labelId] for s in samples]

    return WeightedRandomSampler(
        weights=sampleWeights,
        num_samples=numSamples,
        replacement=True,
    )