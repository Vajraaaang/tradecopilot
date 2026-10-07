"""Training-isolated experimental OHLCV models with non-executable JSON inference artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import sklearn  # type: ignore[import-untyped]
from numpy.typing import NDArray
from pydantic import model_validator
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from tradecopilot.forecast.contracts import LABELS, ForecastConfig, ForecastExample, content_hash
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES, OHLCV_FEATURE_VERSION, OhlcvFeatureRecord
from tradecopilot.models import FrozenModel

ModelKind = Literal["ohlcv_logistic", "ohlcv_hist_gradient_boosting"]
HGB_SETTINGS = {
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "max_depth": 3,
    "min_samples_leaf": 100,
    "l2_regularization": 10.0,
    "early_stopping": False,
    "categorical_features": None,
    "class_weight": None,
}


def _matrix(records: Sequence[OhlcvFeatureRecord], symbols: tuple[str, ...]) -> NDArray[np.float64]:
    if any(row.feature_version != OHLCV_FEATURE_VERSION or row.symbol not in symbols for row in records):
        raise ValueError("model feature schema/symbol mismatch")
    return np.asarray(
        [
            [np.nan if row.values[name] is None else row.values[name] for name in OHLCV_FEATURE_NAMES]
            + [float(row.symbol == symbol) for symbol in symbols]
            for row in records
        ],
        dtype=float,
    ).reshape(len(records), len(OHLCV_FEATURE_NAMES) + len(symbols))


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    logits = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


class OhlcvModelArtifact(FrozenModel):
    schema_version: Literal["ohlcv-cpu-model-v2"] = "ohlcv-cpu-model-v2"
    feature_version: Literal["causal-ohlcv-v2"] = "causal-ohlcv-v2"
    name: ModelKind
    config_id: str
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    training_ids: tuple[str, ...]
    train_features_labels_hash: str
    parameters: dict[str, Any]
    sklearn_version: str = sklearn.__version__

    @model_validator(mode="after")
    def valid_model(self) -> Self:
        count = len(OHLCV_FEATURE_NAMES) + len(self.symbols)
        expected = OHLCV_FEATURE_NAMES + tuple(f"symbol_{symbol}" for symbol in self.symbols)
        if self.feature_names != expected or not self.training_ids or not self.config_id:
            raise ValueError("invalid model feature/provenance schema")
        if self.name == "ohlcv_logistic":
            shapes = {"mean": (count,), "scale": (count,), "coefficients": (3, count), "intercept": (3,)}
            for name, shape in shapes.items():
                array = np.asarray(self.parameters.get(name), dtype=float)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError("invalid logistic model parameters")
            if (np.asarray(self.parameters["scale"]) <= 0).any():
                raise ValueError("invalid model scale")
        else:
            baseline = np.asarray(self.parameters.get("baseline"), dtype=float)
            trees = self.parameters.get("trees")
            if baseline.shape != (3,) or not np.isfinite(baseline).all() or not isinstance(trees, list) or not trees:
                raise ValueError("invalid histogram tree model")
            if len(trees) % 3 or len(trees) > 300:
                raise ValueError("invalid histogram tree count")
            version = self.parameters.get("tree_format")
            if not isinstance(version, str) or version not in {
                "numeric_hist_predictor_nodes_v1",
                "numeric_hist_predictor_nodes_v2",
            }:
                raise ValueError("unsupported histogram tree format")
            for tree in trees:
                if not isinstance(tree, list) or not tree or len(tree) > 63:
                    raise ValueError("invalid histogram tree size")
                for index, node in enumerate(tree):
                    keys = {
                        "leaf",
                        "value",
                        "feature",
                        "threshold",
                        "left",
                        "right",
                        "missing_left",
                    }
                    if version == "numeric_hist_predictor_nodes_v2":
                        keys.add("missing_only")
                    if not isinstance(node, dict) or set(node) != keys:
                        raise ValueError("invalid histogram tree node schema")
                    if (
                        not isinstance(node["leaf"], bool)
                        or not isinstance(node["missing_left"], bool)
                        or not isinstance(node.get("missing_only", False), bool)
                        or (node.get("missing_only", False) and (node["leaf"] or node["threshold"] != 0))
                        or not isinstance(node["value"], (int, float))
                        or isinstance(node["value"], bool)
                        or not np.isfinite(node["value"])
                        or not isinstance(node["threshold"], (int, float))
                        or not np.isfinite(node["threshold"])
                    ):
                        raise ValueError("invalid histogram tree value")
                    if not node["leaf"] and (
                        not isinstance(node["feature"], int)
                        or isinstance(node["feature"], bool)
                        or not 0 <= node["feature"] < count
                        or any(
                            not isinstance(node[child], int)
                            or isinstance(node[child], bool)
                            or not index < node[child] < len(tree)
                            for child in ("left", "right")
                        )
                    ):
                        raise ValueError("invalid or cyclic histogram tree pointer")
        return self

    @property
    def model_id(self) -> str:
        return self.name + "-" + content_hash(self.model_dump(mode="json"))[:16]

    def training_matrix(self, records: Sequence[OhlcvFeatureRecord]) -> NDArray[np.float64]:
        if any(row.config_id != self.config_id for row in records):
            raise ValueError("model and feature target config do not match")
        matrix = _matrix(records, self.symbols)
        if self.name == "ohlcv_logistic":
            means = np.asarray(self.parameters["mean"])
            matrix = np.where(np.isnan(matrix), means, matrix)
            matrix = (matrix - means) / np.asarray(self.parameters["scale"])
        return matrix

    def probabilities(self, records: Sequence[OhlcvFeatureRecord]) -> NDArray[np.float64]:
        if not records:
            return np.empty((0, 3), dtype=float)
        matrix = self.training_matrix(records)
        if self.name == "ohlcv_logistic":
            logits = matrix @ np.asarray(self.parameters["coefficients"]).T + np.asarray(self.parameters["intercept"])
        else:
            logits = np.tile(np.asarray(self.parameters["baseline"]), (len(matrix), 1))
            for index, tree in enumerate(self.parameters["trees"]):
                pending = [(0, np.arange(len(matrix)))]
                while pending:
                    position, members = pending.pop()
                    if not len(members):
                        continue
                    node = tree[position]
                    if node["leaf"]:
                        logits[members, index % 3] += node["value"]
                    else:
                        values = matrix[members, node["feature"]]
                        finite_left = True if node.get("missing_only", False) else values <= node["threshold"]
                        left = np.where(np.isnan(values), node["missing_left"], finite_left)
                        pending.extend([(node["left"], members[left]), (node["right"], members[~left])])
        return _softmax(logits)


def fit_ohlcv_model(
    kind: ModelKind,
    examples: Sequence[ForecastExample],
    records: Sequence[OhlcvFeatureRecord],
    config: ForecastConfig,
    *, settings: dict[str, float] | None = None,
) -> tuple[OhlcvModelArtifact, Any]:
    overrides = dict(settings or {})
    allowed = {"C"} if kind == "ohlcv_logistic" else {"l2_regularization"}
    if set(overrides) - allowed or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not np.isfinite(value) or not 0.001 <= value <= 100
        for value in overrides.values()
    ):
        raise ValueError("unsupported candidate settings")
    if (
        not examples
        or len(examples) != len(records)
        or any(
            row.example_id != record.base_example_id
            or row.as_of != record.as_of
            or row.symbol != record.symbol
            or row.config_id != config.config_id
            or record.config_id != config.config_id
            or row.label is None
            or row.provenance != "historical"
            for row, record in zip(examples, records, strict=True)
        )
    ):
        raise ValueError("training examples and feature records must align with the frozen historical contract")
    if len({row.example_id for row in examples}) != len(examples):
        raise ValueError("duplicate training examples")
    symbols = tuple(sorted(config.symbols))
    matrix = _matrix(records, symbols)
    targets = np.asarray([LABELS.index(str(row.label)) for row in examples], dtype=np.int64)
    if set(targets.tolist()) != {0, 1, 2}:
        raise ValueError("training requires all three classes")
    parameters: dict[str, Any]
    if kind == "ohlcv_logistic":
        valid = np.isfinite(matrix)
        count = valid.sum(axis=0)
        means = np.divide(np.where(valid, matrix, 0).sum(axis=0), count, out=np.zeros(matrix.shape[1]), where=count > 0)
        imputed = np.where(valid, matrix, means)
        scale = imputed.std(axis=0)
        scale[scale < 1e-12] = 1
        classifier = LogisticRegression(
            C=overrides.get("C", 1.0), class_weight=None, max_iter=1000, solver="lbfgs", random_state=config.seed
        )
        classifier.fit((imputed - means) / scale, targets)
        parameters = {
            "mean": means.tolist(),
            "scale": scale.tolist(),
            "coefficients": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist(),
            "fit_settings": {"C": overrides.get("C", 1.0), "class_weight": None,
                             "max_iter": 1000, "solver": "lbfgs", "seed": config.seed},
            "all_training_missing_columns": [int(index) for index in np.flatnonzero(count == 0)],
        }
    elif kind == "ohlcv_hist_gradient_boosting":
        fit_settings = HGB_SETTINGS | overrides
        classifier = HistGradientBoostingClassifier(**fit_settings, random_state=config.seed)
        classifier.fit(matrix, targets)
        trees = []
        for iteration in classifier._predictors:
            for predictor in iteration:
                if predictor.nodes["is_categorical"].any():
                    raise ValueError("JSON artifact supports numeric trees only")
                serialized = []
                for node in predictor.nodes:
                    threshold = float(node["num_threshold"])
                    if not np.isfinite(threshold) and not np.isposinf(threshold):
                        raise ValueError("unsupported histogram threshold")
                    missing_only = not bool(node["is_leaf"]) and bool(np.isposinf(threshold))
                    serialized.append(
                        {
                            "leaf": bool(node["is_leaf"]),
                            "value": float(node["value"]),
                            "feature": int(node["feature_idx"]),
                            "threshold": 0.0 if np.isposinf(threshold) else threshold,
                            "left": int(node["left"]),
                            "right": int(node["right"]),
                            "missing_left": bool(node["missing_go_to_left"]),
                            "missing_only": missing_only,
                        }
                    )
                trees.append(serialized)
        parameters = {
            "baseline": np.asarray(classifier._baseline_prediction).ravel().tolist(),
            "trees": trees,
            "tree_format": "numeric_hist_predictor_nodes_v2",
            "fit_settings": fit_settings | {"seed": config.seed},
        }
    else:
        raise ValueError("unknown OHLCV model")
    artifact = OhlcvModelArtifact(
        name=kind,
        config_id=config.config_id,
        symbols=symbols,
        feature_names=OHLCV_FEATURE_NAMES + tuple(f"symbol_{symbol}" for symbol in symbols),
        training_ids=tuple(row.example_id for row in examples),
        train_features_labels_hash=content_hash(
            {"features": [row.model_dump(mode="json") for row in records], "labels": targets.tolist()}
        ),
        parameters=parameters,
    )
    return artifact, classifier


def save_ohlcv_model(model: OhlcvModelArtifact, path: Path) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_ohlcv_model(path: Path) -> OhlcvModelArtifact:
    return OhlcvModelArtifact.model_validate_json(path.resolve().read_text(encoding="utf-8"))
