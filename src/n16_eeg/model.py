"""Leakage-safe NumPy logistic baseline and model persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FoldMetrics:
    subject: str
    balanced_accuracy: float
    macro_f1: float
    roc_auc: float | None
    samples: int


@dataclass
class LogisticModel:
    """Serializable binary logistic model with fitted standardization."""

    feature_names: list[str]
    mean: list[float]
    scale: list[float]
    weights: list[float]
    bias: float
    threshold: float = 0.5
    model_type: str = "numpy_logistic_regression"
    format_version: int = 1

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {array.shape[1]}"
            )
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        z = ((array - mean) / scale) @ weights + self.bias
        z = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_feature_dict(self, features: dict[str, float]) -> float:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing model features: {missing}")
        row = np.array([features[name] for name in self.feature_names], dtype=float)
        return float(self.predict_proba(row)[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LogisticModel":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_type") != "numpy_logistic_regression":
            raise ValueError("Unsupported model type")
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported model format version")
        return cls(**payload)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logistic_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    learning_rate: float = 0.08,
    epochs: int = 1600,
    l2: float = 0.01,
) -> LogisticModel:
    """Fit class-balanced logistic regression with deterministic gradient descent."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2-D and have the same row count as y")
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match X columns")
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Training data must contain labels 0 and 1")
    if not np.all(np.isfinite(X)):
        raise ValueError("Training features contain non-finite values")

    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0
    standardized = (X - mean) / scale
    weights = np.zeros(X.shape[1], dtype=float)
    bias = 0.0

    positives = max(float(np.sum(y == 1)), 1.0)
    negatives = max(float(np.sum(y == 0)), 1.0)
    sample_weight = np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives))
    weight_sum = float(np.sum(sample_weight))

    for epoch in range(epochs):
        probability = _sigmoid(standardized @ weights + bias)
        error = (probability - y) * sample_weight
        gradient_w = (standardized.T @ error) / weight_sum + l2 * weights
        gradient_b = float(np.sum(error) / weight_sum)
        step = learning_rate / (1.0 + 0.001 * epoch)
        weights -= step * gradient_w
        bias -= step * gradient_b

    return LogisticModel(
        feature_names=list(feature_names),
        mean=mean.tolist(),
        scale=scale.tolist(),
        weights=weights.tolist(),
        bias=float(bias),
    )


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(np.sum(ranks[y_true == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _binary_metrics(
    y_true: np.ndarray, probability: np.ndarray
) -> tuple[float, float, float | None]:
    y_true = np.asarray(y_true, dtype=int)
    predicted = (np.asarray(probability) >= 0.5).astype(int)
    tp = int(np.sum((predicted == 1) & (y_true == 1)))
    tn = int(np.sum((predicted == 0) & (y_true == 0)))
    fp = int(np.sum((predicted == 1) & (y_true == 0)))
    fn = int(np.sum((predicted == 0) & (y_true == 1)))
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    balanced_accuracy = (tpr + tnr) / 2.0
    f1_positive = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_negative = 2 * tn / max(2 * tn + fp + fn, 1)
    macro_f1 = (f1_positive + f1_negative) / 2.0
    return balanced_accuracy, macro_f1, _roc_auc(y_true, np.asarray(probability, dtype=float))


def evaluate_loso(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    feature_names: list[str] | None = None,
) -> list[FoldMetrics]:
    """Evaluate a fresh scaler/model on every held-out-subject fold."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    subjects = np.asarray(subjects).astype(str)
    if not (len(X) == len(y) == len(subjects)):
        raise ValueError("X, y and subjects must have the same number of rows")
    names = feature_names or [f"feature_{index}" for index in range(X.shape[1])]

    results: list[FoldMetrics] = []
    for held_out in np.unique(subjects):
        test = subjects == held_out
        train = ~test
        if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
            continue
        estimator = fit_logistic_model(X[train], y[train], names)
        probability = estimator.predict_proba(X[test])
        balanced_accuracy, macro_f1, auc = _binary_metrics(y[test], probability)
        results.append(
            FoldMetrics(
                subject=held_out,
                balanced_accuracy=float(balanced_accuracy),
                macro_f1=float(macro_f1),
                roc_auc=float(auc) if auc is not None else None,
                samples=int(np.sum(test)),
            )
        )
    if not results:
        raise ValueError("No valid LOSO fold contains both classes in train and test")
    return results
