# src/validation/__init__.py
"""Input validation và OOD detection cho PlantDocAI."""

from .ood_detector import (
    EnergyOODDetector,
    OODResult,
    compute_energy_score,
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_TEMPERATURE,
)
