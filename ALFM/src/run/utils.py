"""Utilities for Active Learning experiments."""

from typing import Dict

import numpy as np
from ALFM.src.run.logging import logger
from numpy.typing import NDArray


def log_composition(
    features: NDArray[np.float32], labels: NDArray[np.int64], mask: NDArray[np.bool_]
) -> None:
    total_samples = len(features)
    num_samples = len(features[mask])
    num_classes = len(np.unique(labels))
    seen_classes = len(np.unique(labels[mask]))
    num_features = features.shape[1]

    logger.info(
        f"Training on {num_samples}/{total_samples} samples with dim: "
        + f"{num_features}, seen {seen_classes}/{num_classes} classes"
    )


def log_scores(scores: Dict[str, float], i: int, num_iter: int, budget: int) -> None:
    logger.info(
        f"[{i}/{num_iter}] Budget: {budget} "
        + f"| Acc: {scores['TEST_MulticlassAccuracy']:.4f}"
        + f" | AUROC: {scores['TEST_MulticlassAUROC']:.4f}"
    )
