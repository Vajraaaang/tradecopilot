from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import sklearn  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pydantic import Field, model_validator
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from tradecopilot.forecast.contracts import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    LABELS,
    ForecastConfig,
    ForecastExample,
    ForecastPrediction,
    content_hash,
)
from tradecopilot.models import FrozenModel


class ModelArtifact(FrozenModel):
    name: str
    config_id: str
    training_ids: tuple[str, ...]
    parameters: dict[str, Any]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    feature_version: str = FEATURE_VERSION
    validation_ids: tuple[str, ...] = ()
    temperature: float = Field(default=1, gt=0, le=10, allow_inf_nan=False)
    schema_version: str = "cpu-baseline-v1"
    sklearn_version: str = sklearn.__version__

    @model_validator(mode="after")
    def valid_artifact(self) -> Self:
        if self.feature_names != FEATURE_NAMES or self.feature_version != FEATURE_VERSION:
            raise ValueError("unsupported model feature schema")
        if self.name not in {"prior", "momentum", "logistic"}:
            raise ValueError("unknown baseline model")
        if not self.training_ids or not self.config_id:
            raise ValueError("model needs training/config provenance")
        if self.name == "logistic":
            shapes = {
                "mean": (len(FEATURE_NAMES),),
                "scale": (len(FEATURE_NAMES),),
                "coefficients": (len(LABELS), len(FEATURE_NAMES)),
                "intercept": (len(LABELS),),
            }
            for key, shape in shapes.items():
                array = np.asarray(self.parameters.get(key), dtype=float)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError("invalid logistic artifact")
            if (np.asarray(self.parameters["scale"]) <= 0).any():
                raise ValueError("model scale must be positive")
        else:
            if self.name == "momentum":
                band = self.parameters.get("band")
                if not isinstance(band, (int, float)) or isinstance(band, bool) or not np.isfinite(band) or band < 0:
                    raise ValueError("invalid momentum band")
            rows = [self.parameters.get("prior")] if self.name == "prior" else self.parameters.get("bins", [])
            expected = 1 if self.name == "prior" else 3
            array = np.asarray(rows, dtype=float)
            if (
                array.shape != (expected, 3)
                or not np.isfinite(array).all()
                or (array < 0).any()
                or not np.allclose(array.sum(axis=1), 1)
            ):
                raise ValueError("invalid baseline probabilities")
        return self

    @property
    def model_id(self) -> str:
        return self.name + "-" + content_hash(self.model_dump(mode="json"))[:16]

    def probabilities(self, examples: Sequence[ForecastExample]) -> NDArray[np.float64]:
        if any(example.config_id != self.config_id for example in examples):
            raise ValueError("model and example config do not match")
        if not examples:
            return np.empty((0, 3), dtype=float)
        matrix = np.asarray([[row.features[name] for name in FEATURE_NAMES] for row in examples], dtype=float)
        if self.name == "prior":
            probabilities = np.tile(np.asarray(self.parameters["prior"], dtype=float), (len(examples), 1))
        elif self.name == "momentum":
            band = float(self.parameters["band"])
            indices = np.where(matrix[:, 1] < -band, 0, np.where(matrix[:, 1] > band, 2, 1))
            probabilities = np.asarray(self.parameters["bins"], dtype=float)[indices]
        else:
            normalized = (matrix - np.asarray(self.parameters["mean"])) / np.asarray(self.parameters["scale"])
            logits = normalized @ np.asarray(self.parameters["coefficients"]).T + np.asarray(
                self.parameters["intercept"]
            )
            probabilities = _softmax(logits)
        return _softmax(np.log(np.clip(probabilities, 1e-12, 1)) / self.temperature)


def _softmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    values = values - values.max(axis=1, keepdims=True)
    weights = np.exp(values)
    return weights / weights.sum(axis=1, keepdims=True)


def fit_model(
    name: str,
    examples: Sequence[ForecastExample],
    config: ForecastConfig,
    *,
    class_weight: Literal["balanced"] | None = "balanced",
) -> ModelArtifact:
    if not examples or any(row.label is None or row.config_id != config.config_id for row in examples):
        raise ValueError("training requires labeled examples matching the config")
    targets = np.asarray([LABELS.index(str(row.label)) for row in examples], dtype=np.int64)
    matrix = np.asarray([[row.features[key] for key in FEATURE_NAMES] for row in examples], dtype=float)
    counts = np.bincount(targets, minlength=3).astype(float) + 1  # fixed Laplace smoothing
    parameters: dict[str, Any]
    if name == "prior":
        parameters = {"prior": (counts / counts.sum()).tolist()}
    elif name == "momentum":
        bins = np.ones((3, 3), dtype=float)
        for row, label in zip(examples, targets, strict=True):
            change = row.features["return_5m_bps"]
            bucket = 0 if change < -config.flat_threshold_bps else 2 if change > config.flat_threshold_bps else 1
            bins[bucket, int(label)] += 1
        parameters = {"bins": (bins / bins.sum(axis=1, keepdims=True)).tolist(), "band": config.flat_threshold_bps}
    elif name == "logistic":
        if len(set(targets.tolist())) != 3:
            raise ValueError("logistic baseline requires all three classes in the training period")
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-12] = 1
        classifier = LogisticRegression(
            max_iter=1000, random_state=config.seed, class_weight=class_weight, solver="lbfgs"
        )
        classifier.fit((matrix - mean) / scale, targets)
        parameters = {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "coefficients": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist(),
            "fit_settings": {
                "C": 1.0,
                "class_weight": class_weight,
                "solver": "lbfgs",
                "max_iter": 1000,
                "seed": config.seed,
            },
        }
    else:
        raise ValueError("unknown baseline model")
    return ModelArtifact(
        name=name,
        config_id=config.config_id,
        training_ids=tuple(row.example_id for row in examples),
        parameters=parameters,
    )


def calibrate_model(model: ModelArtifact, validation: Sequence[ForecastExample]) -> ModelArtifact:
    if not validation or any(row.label is None for row in validation):
        raise ValueError("calibration requires labeled validation data")
    if set(model.training_ids) & {row.example_id for row in validation}:
        raise ValueError("calibration data must be separate from training")
    probabilities = model.model_copy(update={"temperature": 1.0}).probabilities(validation)
    targets = np.asarray([LABELS.index(str(row.label)) for row in validation], dtype=np.int64)
    candidates = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    losses = []
    for temperature in candidates:
        calibrated = _softmax(np.log(np.clip(probabilities, 1e-12, 1)) / temperature)
        losses.append(float(-np.log(np.clip(calibrated[np.arange(len(targets)), targets], 1e-12, 1)).mean()))
    selected = candidates[int(np.argmin(losses))]
    return model.model_copy(
        update={"temperature": selected, "validation_ids": tuple(row.example_id for row in validation)}
    )


def predict_model(
    model: ModelArtifact,
    examples: Sequence[ForecastExample],
    dataset_id: str,
    config: ForecastConfig,
) -> list[ForecastPrediction]:
    if model.config_id != config.config_id:
        raise ValueError("model config does not match prediction config")
    predictions = []
    model_id = model.model_id
    for example in examples:
        started = time.perf_counter()
        values = model.probabilities([example])[0]
        latency = (time.perf_counter() - started) * 1000
        confidence = float(max(values))
        abstained = confidence < config.abstention_threshold
        predictions.append(
            ForecastPrediction(
                example_id=example.example_id,
                dataset_id=dataset_id,
                model_id=model_id,
                generated_at=datetime.now(UTC),
                status="abstained" if abstained else "ok",
                execution="local",
                probabilities={label: float(value) for label, value in zip(LABELS, values, strict=True)},
                reason="below_probability_threshold" if abstained else None,
                model_version=model.schema_version,
                calibration_version="temperature-v1" if model.validation_ids else None,
                model_confidence=confidence,
                latency_ms=latency,
                estimated_cost_usd=0,
            )
        )
    return predictions


def save_model(model: ModelArtifact, path: Path) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_model(path: Path) -> ModelArtifact:
    # JSON data only: loading model artifacts never executes pickled code.
    return ModelArtifact.model_validate_json(path.resolve().read_text(encoding="utf-8"))
