import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.training.checkpoint import saveCheckpoint
from src.training.metrics import AverageMeter, accuracyAtK
from src.models.modelFactory import unfreezeModel


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        trainLoader: DataLoader,
        valLoader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        classNames: list[str],
        outputDir: str,
        topK: int = 3,
        saveBestMetric: str = "valTop1",
        # ── Staged fine-tuning ────────────────────────────────────────────────
        useStagedFinetuning: bool = False,
        headOnlyEpochs: int = 3,
        # ── Optimizer rebuild config (needed for staged unfreeze) ─────────────
        optimizerLR: float = 0.001,
        weightDecay: float = 0.0001,
        backboneLR: Optional[float] = None,
        useLayerLR: bool = False,
        stageUnfreezeApplied: bool = False,
    ):
        self.model = model
        self.device = device
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.classNames = classNames
        self.outputDir = Path(outputDir)
        self.topK = topK
        self.saveBestMetric = saveBestMetric

        # Staged fine-tuning state
        self.useStagedFinetuning = useStagedFinetuning
        self.headOnlyEpochs = headOnlyEpochs
        self._stageUnfreezeApplied = stageUnfreezeApplied  # Khởi tạo từ parameter

        # Lưu optimizer config để rebuild sau staged unfreeze
        self._optimizerLR = optimizerLR
        self._weightDecay = weightDecay
        self._backboneLR = backboneLR
        self._useLayerLR = useLayerLR

        self.checkpointDir = self.outputDir / "checkpoints"
        self.logDir = self.outputDir / "logs"
        self.checkpointDir.mkdir(parents=True, exist_ok=True)
        self.logDir.mkdir(parents=True, exist_ok=True)

        self.csvLogPath = self.logDir / "trainMetrics.csv"
        self.bestMetric = float("-inf")

        self._initCsvLogger()

    def _initCsvLogger(self):
        if not self.csvLogPath.exists():
            with open(self.csvLogPath, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "epoch",
                    "trainLoss",
                    "trainTop1",
                    "trainTopK",
                    "valLoss",
                    "valTop1",
                    "valTopK",
                    "lr",
                    "epochSeconds",
                ])

    def _writeCsvRow(self, rowData: Dict[str, Any]):
        with open(self.csvLogPath, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                rowData["epoch"],
                f"{rowData['trainLoss']:.6f}",
                f"{rowData['trainTop1']:.6f}",
                f"{rowData['trainTopK']:.6f}",
                f"{rowData['valLoss']:.6f}",
                f"{rowData['valTop1']:.6f}",
                f"{rowData['valTopK']:.6f}",
                f"{rowData['lr']:.8f}",
                f"{rowData['epochSeconds']:.2f}",
            ])

    def _runOneEpoch(self, loader: DataLoader, training: bool):
        if training:
            self.model.train()
        else:
            self.model.eval()

        lossMeter = AverageMeter()
        top1Meter = AverageMeter()
        topKMeter = AverageMeter()

        validBatchCount = 0

        for batch in loader:
            if batch is None:
                continue

            images, targets = batch
            if images.size(0) == 0:
                continue

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(training):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            batchMetrics = accuracyAtK(logits, targets, topK=self.topK)
            batchSize = images.size(0)

            lossMeter.update(loss.item(), batchSize)
            top1Meter.update(batchMetrics["top1"], batchSize)
            topKMeter.update(batchMetrics["topK"], batchSize)
            validBatchCount += 1

        if validBatchCount == 0:
            raise RuntimeError(
                "No valid batch found. Check dataset path, corrupt images, or collateFn filtering."
            )

        return {
            "loss": lossMeter.avg,
            "top1": top1Meter.avg,
            "topK": topKMeter.avg,
        }

    def _countTrainableParams(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _applyStageUnfreeze(self, config: Dict[str, Any]) -> None:
        """
        Unfreeze toàn bộ backbone và tạo lại optimizer với LR thấp hơn.

        Lý do tạo lại optimizer thay vì add_param_group():
          - Khi staged unfreeze xảy ra, backbone lần đầu tiên trở thành trainable.
          - Adam optimizer cũ không có momentum history cho backbone params.
          - Tạo lại optimizer sạch đảm bảo tất cả param groups được quản lý
            đúng ngay từ đầu giai đoạn 2, tránh lỗi tiềm ẩn.
          - LR được giảm 10x để tránh shock gradient khi fine-tune backbone.
        """
        print("\n" + "="*60)
        print("[STAGED] Giai đoạn 1 kết thúc. Bắt đầu Giai đoạn 2: Unfreeze backbone.")
        paramsBefore = self._countTrainableParams()
        print(f"[STAGED] Trainable params trước unfreeze: {paramsBefore:,}")

        # Unfreeze toàn bộ model
        unfreezeModel(self.model)

        paramsAfter = self._countTrainableParams()
        print(f"[STAGED] Trainable params sau unfreeze:  {paramsAfter:,}")

        # LR giai đoạn 2: giảm 10x để tránh catastrophic forgetting
        stage2LR = self._optimizerLR * 0.1
        stage2BackboneLR = (self._backboneLR * 0.1) if self._backboneLR is not None else stage2LR

        if self._useLayerLR:
            from src.models.modelFactory import buildParamGroups
            paramGroups = buildParamGroups(
                model=self.model,
                headLR=stage2LR,
                backboneLR=stage2BackboneLR,
            )
        else:
            trainableParams = [p for p in self.model.parameters() if p.requires_grad]
            paramGroups = [{"params": trainableParams, "lr": stage2LR}]
            print(f"[STAGED] Optimizer rebuilt: single group — {len(trainableParams)} params @ lr={stage2LR:.2e}")

        # Tạo lại optimizer Adam với cùng weight_decay
        self.optimizer = torch.optim.Adam(paramGroups, weight_decay=self._weightDecay)
        print(f"[STAGED] Optimizer Adam rebuilt cho giai đoạn 2 (stage2LR={stage2LR:.2e})")
        print("="*60 + "\n")

        self._stageUnfreezeApplied = True

    def fit(self, numEpochs: int, config: Dict[str, Any], startEpoch: int = 1):
        print("[INFO] Start training loop")

        # Nếu resume vào giai đoạn 2 (epoch > headOnlyEpochs), backbone đã được unfreeze
        # trong checkpoint → không cần unfreeze lại, chỉ cần đánh dấu đã apply.
        if self.useStagedFinetuning and startEpoch > self.headOnlyEpochs:
            self._stageUnfreezeApplied = True
            print(
                f"[STAGED] Resume từ epoch {startEpoch} — đã qua giai đoạn 1 "
                f"(headOnlyEpochs={self.headOnlyEpochs}). Backbone được xem là đã unfreeze."
            )

        for epoch in range(startEpoch, numEpochs + 1):
            # ── Staged unfreeze check ──────────────────────────────────────────
            if (
                self.useStagedFinetuning
                and not self._stageUnfreezeApplied
                and epoch == self.headOnlyEpochs + 1
            ):
                self._applyStageUnfreeze(config)

            startTime = time.time()

            trainMetrics = self._runOneEpoch(self.trainLoader, training=True)
            valMetrics = self._runOneEpoch(self.valLoader, training=False)

            if self.scheduler is not None:
                self.scheduler.step()

            currentLr = self.optimizer.param_groups[0]["lr"]
            epochSeconds = time.time() - startTime

            rowData = {
                "epoch": epoch,
                "trainLoss": trainMetrics["loss"],
                "trainTop1": trainMetrics["top1"],
                "trainTopK": trainMetrics["topK"],
                "valLoss": valMetrics["loss"],
                "valTop1": valMetrics["top1"],
                "valTopK": valMetrics["topK"],
                "lr": currentLr,
                "epochSeconds": epochSeconds,
            }
            self._writeCsvRow(rowData)

            print(
                f"[Epoch {epoch}/{numEpochs}] "
                f"trainLoss={trainMetrics['loss']:.4f} "
                f"trainTop1={trainMetrics['top1']:.4f} "
                f"trainTop{self.topK}={trainMetrics['topK']:.4f} | "
                f"valLoss={valMetrics['loss']:.4f} "
                f"valTop1={valMetrics['top1']:.4f} "
                f"valTop{self.topK}={valMetrics['topK']:.4f} | "
                f"lr={currentLr:.8f}"
            )

            metricValue = rowData[self.saveBestMetric]
            if metricValue > self.bestMetric:
                self.bestMetric = metricValue
                saveCheckpoint(
                    savePath=str(self.checkpointDir / "best.pt"),
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    bestMetric=self.bestMetric,
                    classNames=self.classNames,
                    config=config,
                )
                print(f"[INFO] Saved new best checkpoint: {self.checkpointDir / 'best.pt'}")

            saveCheckpoint(
                savePath=str(self.checkpointDir / "last.pt"),
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                bestMetric=self.bestMetric,
                classNames=self.classNames,
                config=config,
            )