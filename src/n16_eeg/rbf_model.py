"""NumPy inference for a binary RBF SVM trained on log-scaled EEG features."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RbfModel:
    feature_names: list[str]
    center: list[float]
    scale: list[float]
    support_vectors: list[list[float]]
    dual_coefficients: list[float]
    intercept: float
    gamma: float
    threshold: float
    model_type: str = "numpy_rbf_svm"
    format_version: int = 1
    _vectors: np.ndarray = field(init=False, repr=False)
    _coefficients: np.ndarray = field(init=False, repr=False)
    _center: np.ndarray = field(init=False, repr=False)
    _scale: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._center = np.asarray(self.center, dtype=float)
        self._scale = np.asarray(self.scale, dtype=float)
        self._vectors = np.asarray(self.support_vectors, dtype=float)
        self._coefficients = np.asarray(self.dual_coefficients, dtype=float)
        width = len(self.feature_names)
        if (not width or len(set(self.feature_names)) != width
                or self._center.shape != (width,) or self._scale.shape != (width,)
                or self._coefficients.ndim != 1
                or self._vectors.shape != (len(self._coefficients), width)
                or not all(np.isfinite(array).all() for array in
                           (self._center, self._scale, self._vectors, self._coefficients))
                or np.any(self._scale <= 0) or not 0 < self.gamma < np.inf
                or not np.isfinite(self.intercept) or not 0 <= self.threshold <= 1):
            raise ValueError("Invalid RBF model parameters")

    def predict_margin(self, features: np.ndarray) -> float:
        features = np.asarray(features, dtype=float)
        if (features.shape != self._center.shape or not np.isfinite(features).all()
                or np.any(features < 0)):
            raise ValueError("Expected one finite nonnegative feature vector matching the RBF model")
        scaled = (np.log1p(features) - self._center) / self._scale
        differences = self._vectors - scaled
        squared_distance = np.einsum("ij,ij->i", differences, differences)
        kernels = np.exp(-self.gamma * squared_distance)
        return float(np.dot(kernels, self._coefficients) + self.intercept)

    def predict_feature_dict(self, features: dict[str, float]) -> float:
        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise ValueError(f"Missing RBF model features: {missing}")
        row = np.array([features[name] for name in self.feature_names], dtype=float)
        margin = self.predict_margin(row)
        return float(1.0 / (1.0 + np.exp(-np.clip(margin, -30.0, 30.0))))
