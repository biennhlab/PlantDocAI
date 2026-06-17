import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def saveCheckpoint(
    savePath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    bestMetric: float,
    classNames: list[str],
    config: Dict[str, Any],
) -> None:
    """
    Save đầy đủ checkpoint để có thể resume training hoặc evaluate sau này.
    """
    path = Path(savePath)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "bestMetric": bestMetric,
        "modelStateDict": model.state_dict(),
        "optimizerStateDict": optimizer.state_dict(),
        "schedulerStateDict": scheduler.state_dict() if scheduler is not None else None,
        "classNames": classNames,
        "config": config,
    }
    torch.save(checkpoint, path)


def loadCheckpoint(
    checkpointPath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    mapLocation: str = "cpu",
):
    """
    Load checkpoint vào model và tùy chọn load luôn optimizer/scheduler.
    """
    checkpoint = torch.load(checkpointPath, map_location=mapLocation)

    model.load_state_dict(checkpoint["modelStateDict"])

    if optimizer is not None and checkpoint.get("optimizerStateDict") is not None:
        optimizer.load_state_dict(checkpoint["optimizerStateDict"])

    if scheduler is not None and checkpoint.get("schedulerStateDict") is not None:
        scheduler.load_state_dict(checkpoint["schedulerStateDict"])

    return checkpoint


def findCheckpoint(modelDir: str) -> Optional[Path]:
    """
    Tìm checkpoint file theo thứ tự ưu tiên chuẩn.

    Thứ tự: checkpoints/best.pt → best.pt → checkpoints/last.pt.

    Args:
        modelDir: Đường dẫn tới thư mục artifact chứa checkpoint.

    Returns:
        Path tới checkpoint nếu tìm thấy, None nếu không.
    """
    base = Path(modelDir)
    for candidate in ("checkpoints/best.pt", "best.pt", "checkpoints/last.pt"):
        path = base / candidate
        if path.exists():
            return path
    return None


def saveJson(savePath: str, data: Dict[str, Any]):
    """
    Save dictionary ra file JSON.
    """
    path = Path(savePath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        