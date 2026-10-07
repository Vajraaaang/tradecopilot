"""Registered direct classifiers and strictly allowlisted, non-executable JSON weights."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from tradecopilot.forecast.baselines import ModelArtifact, fit_model
from tradecopilot.forecast.contracts import LABELS, ForecastConfig, content_hash
from tradecopilot.forecast.ohlcv_features import OHLCV_FEATURE_NAMES, OhlcvFeatureRecord
from tradecopilot.forecast.ohlcv_models import OhlcvModelArtifact, fit_ohlcv_model

if TYPE_CHECKING:
    from tradecopilot.forecast.supervised_data import SupervisedStage

CPU_VERSION = "supervised-cpu-model-v1"
NEURAL_VERSION = "supervised-neural-model-v1"
CPU_CANDIDATES: dict[str, dict[str, Any]] = {
    "prior": {"key": "prior", "kind": "training_class_prior"},
    "lr-001": {"key": "lr-001", "kind": "ohlcv_logistic", "C": 0.01},
    "lr-1": {"key": "lr-1", "kind": "ohlcv_logistic", "C": 1.0},
    "hgb-31": {
        "key": "hgb-31", "kind": "ohlcv_hist_gradient_boosting", "learning_rate": 0.05,
        "max_iter": 100, "max_leaf_nodes": 31, "max_depth": 6, "min_samples_leaf": 100,
        "l2_regularization": 10.0, "early_stopping": False, "class_weight": None, "seed": 42,
    },
}
NEURAL_CANDIDATES: dict[str, dict[str, Any]] = {
    "tcn-32": {
        "key": "tcn-32", "kind": "tcn", "input_channels": 6, "width": 32, "kernel_size": 3,
        "dilations": [1, 2, 4, 8, 16], "convolutions_per_block": 1, "input_projection": True,
        "causal_left_padding": True, "static_dim": 115, "head_hidden": 32, "dropout": 0.1,
        "readout": "last_timestep",
    },
    "lstm-64": {
        "key": "lstm-64", "kind": "lstm", "input_channels": 6, "hidden_size": 64,
        "n_lstm_layers": 1, "bidirectional": False, "batch_first": True, "static_dim": 115,
        "head_hidden": 32, "dropout": 0.1, "lstm_dropout": 0.0, "readout": "last_hidden",
        "state_reset": "eachindependentwindow",
    },
}
_NEURAL_SETTINGS: dict[str, Any] = {
    "seeds": [42, 43, 44], "device": "cpu", "intraop_threads": 1, "batch_size": 256,
    "max_epochs": 20, "max_seconds_per_seed": 600, "optimizer": "AdamW", "learning_rate": 0.001,
    "weight_decay": 0.0001, "gradient_clip_norm": 1.0, "loss": "unweighted_cross_entropy",
    "shuffle": "TRAINonly; separatefixedgeneratorperseed soarchitecturesseeidenticalbatchorders",
    "drop_last": False, "num_workers": 0,
    "checkpoint_rule": "lowestcompleteTUNEmeanNLL; improvement>1e-4,patience4,earliesttie; noTRAIN+TUNErefit",
    "incomplete_seed": "familyineligible; nopartialensemble",
    "maximum_optimizer_updates": "20*ceil(N_train/256)", "class_order": list(LABELS),
}
_COMMON_KEYS = {
    "schema_version", "candidate_key", "model_id", "prepared_data_id", "registration_id", "registration",
    "config_id", "class_order", "training_ids", "training_ids_hash", "train_features_labels_hash",
    "normalizers", "normalizer_hash", "fit_settings", "source_hashes",
}
_TORCH_CONFIGURED = False


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    """The same code/dependency identity is available to the study selection seal."""
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "src" / "tradecopilot").rglob("*.py"))
    paths += [root / name for name in (
        "uv.lock", "pyproject.toml", "scripts/train_supervised_seed.py",
        "scripts/prepare_supervised_forecast.py", "scripts/run_supervised_forecast_study.py",
    ) if (root / name).is_file()]
    return {str(path.relative_to(root)): _file_hash(path) for path in paths}


def prepared_input_hashes(prepared: Path, registration_path: Path) -> dict[str, str]:
    # Opaque byte hashes do not decode CAL/GATE/TEST inputs or outcomes.
    paths = sorted(path for path in prepared.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in paths):
        raise ValueError("prepared inputs must not be symlinks")
    return {str(path.relative_to(prepared)): _file_hash(path) for path in paths} | {
        "registration_file": _file_hash(registration_path),
    }


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _read_json(path: Path, maximum_bytes: int = 16_000_000) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise ValueError("invalid JSON artifact file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("artifact must be a JSON object")
    # content_hash rejects nonstandard JSON NaN and infinity values.
    content_hash(value)
    return value


def _registration(registration: dict[str, Any]) -> ForecastConfig:
    if (
        registration.get("schema_version") != "supervised-forecast-registration-v1"
        or registration.get("registration_id") != content_hash(
            {key: value for key, value in registration.items() if key != "registration_id"}
        )
        or registration.get("cpu_candidates") != list(CPU_CANDIDATES.values())
        or registration.get("neural_candidates") != list(NEURAL_CANDIDATES.values())
        or registration.get("neural_training") != _NEURAL_SETTINGS
    ):
        raise ValueError("registration hash or declared model settings mismatch")
    config = ForecastConfig.model_validate(registration["forecast_config"])
    if len(config.symbols) != 5 or config.seed != 42 or config.horizon_minutes != 15 or config.flat_threshold_bps != 10:
        raise ValueError("registration target or symbol contract mismatch")
    return config


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _normalizers(value: Any, training_ids_hash: str, count: int) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"sequence", "static", "fit_role", "train_case_ids_hash", "train_raw_inputs_hash",
                          "fit_case_count", "fit_time_positions_count"}
        or value["fit_role"] != "TRAIN"
        or value["train_case_ids_hash"] != training_ids_hash
        or not _is_hash(value["train_raw_inputs_hash"])
        or type(value["fit_case_count"]) is not int or value["fit_case_count"] != count
        or type(value["fit_time_positions_count"]) is not int or value["fit_time_positions_count"] != count * 60
    ):
        raise ValueError("normalizer TRAIN provenance mismatch")
    for name, width in (("sequence", 6), ("static", 55)):
        component = value[name]
        expected = {"mean", "scale"} | ({"binary_indices"} if name == "static" else set())
        if not isinstance(component, dict) or set(component) != expected:
            raise ValueError("normalizer schema mismatch")
        for key in ("mean", "scale"):
            array = np.asarray(component[key], dtype=float)
            if array.shape != (width,) or not np.isfinite(array).all() or (key == "scale" and (array <= 0).any()):
                raise ValueError("invalid normalizer values")
    binary = [index for index, name in enumerate(OHLCV_FEATURE_NAMES) if name.startswith("missing_window_")]
    if value["static"]["binary_indices"] != binary:
        raise ValueError("normalizer binary feature contract mismatch")


def _stage_check(stage: SupervisedStage, config: ForecastConfig, role: str) -> None:
    count = len(stage.case_ids)
    _neural_inputs(stage.sequence, stage.static)
    if (
        stage.role != role or not count or len(set(stage.case_ids)) != count
        or stage.sequence.shape[0] != count or len(stage.examples) != count or len(stage.features) != count
        or stage.targets.shape != (count,) or stage.targets.dtype != np.int64
        or not set(stage.targets.tolist()) <= {0, 1, 2}
        or any(
            row.example_id != case_id or record.base_example_id != case_id or row.config_id != config.config_id
            or record.config_id != config.config_id or row.symbol != record.symbol or row.as_of != record.as_of
            or row.provenance != "historical" or row.label != LABELS[int(target)]
            for row, record, case_id, target in zip(
                stage.examples, stage.features, stage.case_ids, stage.targets, strict=True
            )
        )
    ):
        raise ValueError(f"aligned labeled {role} stage required")


def _bindings(train: SupervisedStage, manifest: dict[str, Any], registration: dict[str, Any]) -> dict[str, Any]:
    config = _registration(registration)
    _stage_check(train, config, "TRAIN")
    if (
        manifest.get("registration_id") != registration["registration_id"]
        or manifest.get("registration") != registration or manifest.get("config_id") != config.config_id
        or manifest.get("forecast_config") != config.model_dump(mode="json")
        or not _is_hash(manifest.get("prepared_data_id"))
        or manifest.get("normalizer_id") != content_hash(manifest["normalizers"])
    ):
        raise ValueError("prepared manifest and registration do not match")
    ids_hash = content_hash(train.case_ids)
    _normalizers(manifest["normalizers"], ids_hash, len(train.case_ids))
    return {
        "prepared_data_id": manifest["prepared_data_id"], "registration_id": registration["registration_id"],
        "registration": registration, "config_id": config.config_id, "class_order": list(LABELS),
        "training_ids": list(train.case_ids), "training_ids_hash": ids_hash,
        "train_features_labels_hash": content_hash({
            "case_ids": train.case_ids,
            "sequence_float32_sha256": hashlib.sha256(train.sequence.tobytes(order="C")).hexdigest(),
            "static_float32_sha256": hashlib.sha256(train.static.tobytes(order="C")).hexdigest(),
            "features": [row.feature_id for row in train.features], "targets": train.targets.tolist(),
        }),
        "normalizers": manifest["normalizers"], "normalizer_hash": manifest["normalizer_id"],
    }


def _validate_common(artifact: dict[str, Any]) -> ForecastConfig:
    if artifact.get("model_id") != content_hash({k: v for k, v in artifact.items() if k != "model_id"}):
        raise ValueError("model artifact hash mismatch")
    config = _registration(artifact["registration"])
    ids = artifact["training_ids"]
    if (
        artifact["registration_id"] != artifact["registration"]["registration_id"]
        or artifact["config_id"] != config.config_id or artifact["class_order"] != list(LABELS)
        or not isinstance(ids, list) or not ids or any(not _is_hash(value) for value in ids)
        or len(set(ids)) != len(ids) or artifact["training_ids_hash"] != content_hash(ids)
        or not _is_hash(artifact["prepared_data_id"]) or not _is_hash(artifact["train_features_labels_hash"])
        or artifact["normalizer_hash"] != content_hash(artifact["normalizers"])
    ):
        raise ValueError("model identity, class order or input binding mismatch")
    _normalizers(artifact["normalizers"], artifact["training_ids_hash"], len(ids))
    if artifact["source_hashes"] != source_hashes():
        raise ValueError("model source/dependency hashes mismatch")
    return config


def _export_hgb(classifier: Any) -> dict[str, Any]:
    trees = []
    for iteration in classifier._predictors:
        for predictor in iteration:
            if predictor.nodes["is_categorical"].any():
                raise ValueError("only numeric tree nodes are exportable")
            serialized = []
            for node in predictor.nodes:
                threshold = float(node["num_threshold"])
                if not np.isfinite(threshold) and not np.isposinf(threshold):
                    raise ValueError("unsupported histogram threshold")
                serialized.append({
                    "leaf": bool(node["is_leaf"]), "value": float(node["value"]),
                    "feature": int(node["feature_idx"]), "threshold": 0.0 if np.isposinf(threshold) else threshold,
                    "left": int(node["left"]), "right": int(node["right"]),
                    "missing_left": bool(node["missing_go_to_left"]),
                    "missing_only": not bool(node["is_leaf"]) and bool(np.isposinf(threshold)),
                })
            trees.append(serialized)
    return {"baseline": np.asarray(classifier._baseline_prediction).ravel().tolist(), "trees": trees,
            "tree_format": "numeric_hist_predictor_nodes_v2"}


def fit_cpu(
    candidate_key: str, train: SupervisedStage, manifest: dict[str, Any], registration: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    if candidate_key not in CPU_CANDIDATES:
        raise ValueError("unknown registered CPU candidate")
    bindings = _bindings(train, manifest, registration)
    config = _registration(registration)
    if set(train.targets.tolist()) != {0, 1, 2}:
        raise ValueError("TRAIN requires all three classes")
    native = None
    model: ModelArtifact | OhlcvModelArtifact
    with threadpool_limits(limits=1):
        if candidate_key == "prior":
            model = fit_model("prior", train.examples, config, class_weight=None)
            settings = {"laplace_smoothing": 1.0, "class_weight": None}
        elif candidate_key.startswith("lr-"):
            model, native = fit_ohlcv_model(
                "ohlcv_logistic", train.examples, train.features, config,
                settings={"C": CPU_CANDIDATES[candidate_key]["C"]},
            )
            settings = model.parameters["fit_settings"]
        else:
            settings = {k: v for k, v in CPU_CANDIDATES[candidate_key].items() if k not in {"key", "kind"}}
            native = HistGradientBoostingClassifier(
                **{k: v for k, v in settings.items() if k != "seed"}, categorical_features=None, random_state=42,
            )
            matrix = np.asarray([
                [np.nan if row.values[name] is None else row.values[name] for name in OHLCV_FEATURE_NAMES]
                + [float(row.symbol == symbol) for symbol in sorted(config.symbols)] for row in train.features
            ], dtype=float)
            native.fit(matrix, train.targets)
            model = OhlcvModelArtifact(
                name="ohlcv_hist_gradient_boosting", config_id=config.config_id, symbols=tuple(sorted(config.symbols)),
                feature_names=OHLCV_FEATURE_NAMES + tuple(f"symbol_{symbol}" for symbol in sorted(config.symbols)),
                training_ids=train.case_ids, train_features_labels_hash=content_hash({
                    "features": [row.model_dump(mode="json") for row in train.features],
                    "labels": train.targets.tolist(),
                }), parameters=_export_hgb(native) | {"fit_settings": settings | {"categorical_features": None}},
            )
            settings = model.parameters["fit_settings"]
    artifact = bindings | {
        "schema_version": CPU_VERSION, "candidate_key": candidate_key, "fit_settings": settings,
        "source_hashes": source_hashes(), "model_artifact": model.model_dump(mode="json"),
    }
    artifact["model_id"] = content_hash(artifact)
    return artifact, native


def _cpu_model(artifact: dict[str, Any]) -> ModelArtifact | OhlcvModelArtifact:
    if set(artifact) != _COMMON_KEYS | {"model_artifact"} or artifact["schema_version"] != CPU_VERSION:
        raise ValueError("unsupported CPU artifact schema")
    config = _validate_common(artifact)
    key = artifact["candidate_key"]
    if key not in CPU_CANDIDATES:
        raise ValueError("unknown CPU artifact candidate")
    model: ModelArtifact | OhlcvModelArtifact
    expected: dict[str, Any]
    if key == "prior":
        model = ModelArtifact.model_validate(artifact["model_artifact"])
        if model.name != "prior" or model.temperature != 1 or model.validation_ids:
            raise ValueError("only the uncalibrated training prior is registered")
        expected = {"laplace_smoothing": 1.0, "class_weight": None}
    else:
        model = OhlcvModelArtifact.model_validate(artifact["model_artifact"])
        if key.startswith("lr-"):
            expected = {"C": CPU_CANDIDATES[key]["C"], "class_weight": None,
                        "max_iter": 1000, "solver": "lbfgs", "seed": 42}
            name = "ohlcv_logistic"
        else:
            expected = {k: v for k, v in CPU_CANDIDATES[key].items() if k not in {"key", "kind"}}
            expected["categorical_features"] = None
            name = "ohlcv_hist_gradient_boosting"
        if model.name != name or model.symbols != tuple(sorted(config.symbols)):
            raise ValueError("CPU architecture/feature contract mismatch")
        if model.parameters["fit_settings"] != expected:
            raise ValueError("CPU model fitting settings mismatch")
    if (artifact["fit_settings"] != expected or model.config_id != artifact["config_id"]
            or list(model.training_ids) != artifact["training_ids"]):
        raise ValueError("CPU fitting provenance mismatch")
    return model


def save_cpu_artifact(artifact: dict[str, Any], path: Path) -> Path:
    _cpu_model(artifact)
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_json(path, artifact)


def load_cpu_artifact(path: Path) -> dict[str, Any]:
    artifact = _read_json(path)
    _cpu_model(artifact)
    return artifact


def predict_cpu(artifact: dict[str, Any], features: Sequence[OhlcvFeatureRecord]) -> NDArray[np.float64]:
    model = _cpu_model(artifact)
    if any(row.config_id != artifact["config_id"] for row in features):
        raise ValueError("CPU prediction input config mismatch")
    if isinstance(model, OhlcvModelArtifact):
        return model.probabilities(features)
    return np.tile(np.asarray(model.parameters["prior"], dtype=np.float64), (len(features), 1))


def native_cpu_probabilities(
    artifact: dict[str, Any], native: Any, features: Sequence[OhlcvFeatureRecord],
) -> NDArray[np.float64]:
    model = _cpu_model(artifact)
    if isinstance(model, ModelArtifact):
        return predict_cpu(artifact, features)
    if native is None or native.classes_.tolist() != [0, 1, 2]:
        raise ValueError("native classifier class order mismatch")
    if not features:
        return np.empty((0, 3), dtype=np.float64)
    with threadpool_limits(limits=1):
        return np.asarray(native.predict_proba(model.training_matrix(features)), dtype=np.float64)


def _torch() -> Any:
    global _TORCH_CONFIGURED
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("supervised neural models require tradecopilot[neural]") from error
    if not _TORCH_CONFIGURED:
        torch.set_num_threads(1)
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        _TORCH_CONFIGURED = True
    return torch


def make_neural_model(candidate_key: str, seed: int) -> Any:
    if candidate_key not in NEURAL_CANDIDATES or isinstance(seed, bool) or seed not in (42, 43, 44):
        raise ValueError("unknown registered neural architecture/seed")
    torch = _torch()
    torch.manual_seed(seed)

    base = torch.nn.Module

    class DirectForecaster(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__()
            if candidate_key == "tcn-32":
                self.projection = torch.nn.Conv1d(6, 32, 1)
                self.blocks = torch.nn.ModuleList([
                    torch.nn.Conv1d(32, 32, 3, dilation=dilation, padding=0) for dilation in (1, 2, 4, 8, 16)
                ])
                width = 32
            else:
                self.lstm = torch.nn.LSTM(6, 64, num_layers=1, batch_first=True, bidirectional=False, dropout=0)
                width = 64
            self.head = torch.nn.Sequential(
                torch.nn.Linear(width + 115, 32), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(32, 3),
            )

        def encode_sequence(self, sequence: Any) -> Any:
            if candidate_key == "lstm-64":
                output, _ = self.lstm(sequence)  # no hidden/cell state enters or survives this window
                return output
            output = self.projection(sequence.transpose(1, 2))
            for block in self.blocks:
                padded = torch.nn.functional.pad(output, (2 * block.dilation[0], 0))
                output = torch.relu(output + block(padded))
            return output.transpose(1, 2)

        def forward(self, sequence: Any, static: Any) -> Any:
            if candidate_key == "lstm-64":
                _, (hidden, _) = self.lstm(sequence)
                encoded = hidden[-1]
            else:
                encoded = self.encode_sequence(sequence)[:, -1]
            return self.head(torch.cat((encoded, static), dim=1))

    return DirectForecaster().to(device="cpu", dtype=torch.float32)


def _neural_inputs(sequence: NDArray[Any], static: NDArray[Any]) -> None:
    if (
        not isinstance(sequence, np.ndarray) or not isinstance(static, np.ndarray)
        or sequence.ndim != 3 or sequence.shape[1:] != (60, 6)
        or static.shape != (sequence.shape[0], 115) or sequence.dtype != np.float32 or static.dtype != np.float32
        or not np.isfinite(sequence).all() or not np.isfinite(static).all()
    ):
        raise ValueError("neural input must be finite float32 [N,60,6] and [N,115]")


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def neural_probabilities(model: Any, sequence: NDArray[np.float32], static: NDArray[np.float32]) -> NDArray[np.float64]:
    _neural_inputs(sequence, static)
    if not len(sequence):
        return np.empty((0, 3), dtype=np.float64)
    torch = _torch()
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(sequence), 256):
            values = model(torch.from_numpy(sequence[start:start + 256].copy()),
                           torch.from_numpy(static[start:start + 256].copy()))
            logits.append(values.detach().cpu().numpy().astype(np.float64))
    matrix = np.concatenate(logits)
    if matrix.shape != (len(sequence), 3) or not np.isfinite(matrix).all():
        raise ValueError("nonfinite neural logits")
    return _softmax(matrix)


def _tensor_shapes(key: str) -> dict[str, tuple[int, ...]]:
    common = {"head.0.weight": (32, 147 if key == "tcn-32" else 179), "head.0.bias": (32,),
              "head.3.weight": (3, 32), "head.3.bias": (3,)}
    if key == "tcn-32":
        return common | {"projection.weight": (32, 6, 1), "projection.bias": (32,)} | {
            f"blocks.{index}.{name}": shape
            for index in range(5) for name, shape in (("weight", (32, 32, 3)), ("bias", (32,)))
        }
    return common | {"lstm.weight_ih_l0": (256, 6), "lstm.weight_hh_l0": (256, 64),
                     "lstm.bias_ih_l0": (256,), "lstm.bias_hh_l0": (256,)}


def _validate_neural(artifact: dict[str, Any]) -> dict[str, NDArray[np.float32]]:
    keys = _COMMON_KEYS | {
        "architecture", "seed", "tune_ids", "tune_ids_hash", "checkpoint", "budget", "tensors", "torch_version",
        "parameter_count", "input_dimensions",
    }
    if set(artifact) != keys or artifact["schema_version"] != NEURAL_VERSION:
        raise ValueError("unsupported neural artifact schema")
    _validate_common(artifact)
    key = artifact["candidate_key"]
    tune_ids = artifact["tune_ids"]
    expected_budget = {"max_epochs": 20, "max_seconds": 600, "batch_size": 256,
                       "max_optimizer_updates": 20 * math.ceil(len(artifact["training_ids"]) / 256)}
    checkpoint = artifact["checkpoint"]
    if (
        key not in NEURAL_CANDIDATES or artifact["architecture"] != NEURAL_CANDIDATES[key]
        or type(artifact["parameter_count"]) is not int
        or artifact["parameter_count"] != (20579 if key == "tcn-32" else 24291)
        or artifact["input_dimensions"] != {"sequence": [60, 6], "static": 115}
        or isinstance(artifact["seed"], bool) or artifact["seed"] not in (42, 43, 44)
        or artifact["fit_settings"] != _NEURAL_SETTINGS or artifact["budget"] != expected_budget
        or artifact["torch_version"] != _torch().__version__
        or not isinstance(tune_ids, list) or not tune_ids or any(not _is_hash(value) for value in tune_ids)
        or len(set(tune_ids)) != len(tune_ids) or set(tune_ids) & set(artifact["training_ids"])
        or artifact["tune_ids_hash"] != content_hash(tune_ids)
        or not isinstance(checkpoint, dict) or set(checkpoint) != {"epoch", "optimizer_updates", "tune_nll"}
        or type(checkpoint["epoch"]) is not int or not 1 <= checkpoint["epoch"] <= 20
        or type(checkpoint["optimizer_updates"]) is not int
        or checkpoint["optimizer_updates"] != checkpoint["epoch"] * math.ceil(len(artifact["training_ids"]) / 256)
        or type(checkpoint["tune_nll"]) not in (int, float)
        or not np.isfinite(checkpoint["tune_nll"]) or checkpoint["tune_nll"] < 0
    ):
        raise ValueError("neural checkpoint/architecture/provenance mismatch")
    shapes = _tensor_shapes(key)
    tensors = artifact["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != set(shapes):
        raise ValueError("neural tensor names mismatch")
    validated = {}
    for name, shape in shapes.items():
        array = np.asarray(tensors[name])
        if array.shape != shape or array.dtype.kind not in {"i", "u", "f"} or not np.isfinite(array).all():
            raise ValueError("neural tensor shape/value mismatch")
        with np.errstate(over="ignore"):
            array = array.astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError("neural tensor exceeds float32 range")
        validated[name] = array
    return validated


class NeuralPredictor:
    def __init__(self, artifact: dict[str, Any]) -> None:
        tensors = _validate_neural(artifact)
        self.artifact = artifact
        self.model_id: str = artifact["model_id"]
        self._model = make_neural_model(artifact["candidate_key"], artifact["seed"])
        torch = _torch()
        self._model.load_state_dict({name: torch.from_numpy(value) for name, value in tensors.items()}, strict=True)
        self._model.eval()

    def probabilities(self, sequence: NDArray[np.float32], static: NDArray[np.float32]) -> NDArray[np.float64]:
        return neural_probabilities(self._model, sequence, static)


def load_neural_artifact(path: Path) -> NeuralPredictor:
    try:
        return NeuralPredictor(_read_json(path))
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("invalid neural artifact") from error


def _load_stage(directory: Path, role: str) -> tuple[SupervisedStage, dict[str, Any]]:
    from tradecopilot.forecast.supervised_data import load_stage

    return load_stage(directory, role)


def _tune_loss(model: Any, tune: SupervisedStage, started: float) -> float | None:
    torch = _torch()
    total = 0.0
    model.eval()
    with torch.no_grad():
        for first in range(0, len(tune.case_ids), 256):
            if time.monotonic() - started >= 600:
                return None
            last = first + 256
            logits = model(torch.from_numpy(tune.sequence[first:last].copy()),
                           torch.from_numpy(tune.static[first:last].copy())).detach().numpy().astype(np.float64)
            if not np.isfinite(logits).all():
                raise ValueError("nonfinite complete TUNE logits")
            maximum = logits.max(axis=1)
            normalizer = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
            total += float((normalizer - logits[np.arange(len(logits)), tune.targets[first:last]]).sum())
    return total / len(tune.case_ids) if time.monotonic() - started < 600 else None


def train_neural_seed(
    prepared_dir: Path, registration_path: Path, candidate_key: str, seed: int, output_dir: Path,
) -> dict[str, Any]:
    """TRAIN optimizer updates, complete TUNE checkpoints, and a budget including checkpoint I/O."""
    started = time.monotonic()
    prepared_dir = prepared_dir.absolute()
    registration_path = registration_path.absolute()
    output_dir = output_dir.absolute()
    if output_dir == prepared_dir or prepared_dir in output_dir.parents:
        raise ValueError("training outputs must be outside the frozen prepared inputs")
    output_dir.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    inputs_before = prepared_input_hashes(prepared_dir, registration_path)
    result: dict[str, Any] = {
        "schema_version": "supervised-seed-result-v1", "candidate_key": candidate_key, "seed": seed,
        "status": "error", "eligible": False, "model_path": None, "checkpoint_hash": None,
        "checkpoint_file_sha256": None, "epochs_completed": 0, "optimizer_updates": 0,
        "best_epoch": None, "best_tune_nll": None, "source_hashes_before": before,
        "prepared_input_hashes_before": inputs_before, "checkpoints": [], "epoch_history": [],
        "prepared_data_id": None, "registration_id": None, "config_id": None, "normalizer_hash": None,
        "train_case_ids_hash": None, "tune_case_ids_hash": None, "budget": None,
        "parameter_count": None, "input_dimensions": {"sequence": [60, 6], "static": 115},
    }
    try:
        registration = _read_json(registration_path)
        config = _registration(registration)
        if candidate_key not in NEURAL_CANDIDATES or isinstance(seed, bool) or seed not in (42, 43, 44):
            raise ValueError("unknown registered neural candidate/seed")
        train, manifest = _load_stage(prepared_dir, "TRAIN")
        tune, tune_manifest = _load_stage(prepared_dir, "TUNE")
        bindings = _bindings(train, manifest, registration)
        _stage_check(tune, config, "TUNE")
        if (
            manifest != tune_manifest or set(train.case_ids) & set(tune.case_ids)
            or set(train.targets.tolist()) != {0, 1, 2}
            or max(row.label_observed_at for row in train.examples if row.label_observed_at is not None)
            >= min(row.as_of for row in tune.examples)
        ):
            raise ValueError("TRAIN/TUNE provenance, class support or outcome purge mismatch")
        budget = {"max_epochs": 20, "max_seconds": 600, "batch_size": 256,
                  "max_optimizer_updates": 20 * math.ceil(len(train.case_ids) / 256)}
        result.update({key: bindings[key] for key in (
            "prepared_data_id", "registration_id", "config_id", "normalizer_hash",
        )})
        result.update({"train_case_ids_hash": bindings["training_ids_hash"],
                       "tune_case_ids_hash": content_hash(tune.case_ids), "budget": budget})
        torch = _torch()
        model = make_neural_model(candidate_key, seed)
        result["parameter_count"] = sum(int(value.numel()) for value in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_fn = torch.nn.CrossEntropyLoss()
        shuffle_rng = torch.Generator(device="cpu").manual_seed(seed)
        train_sequence = torch.from_numpy(train.sequence.copy())
        train_static = torch.from_numpy(train.static.copy())
        train_targets = torch.from_numpy(train.targets.copy())
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_sequence, train_static, train_targets), batch_size=256,
            shuffle=True, generator=shuffle_rng, drop_last=False, num_workers=0,
        )
        stale_epochs = 0
        best_loss = math.inf
        result["status"] = "complete"
        for epoch in range(1, 21):
            model.train()
            for sequence, static, targets in loader:
                if time.monotonic() - started >= 600:
                    result["status"] = "time_cap"
                    break
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(sequence, static), targets)
                if not np.isfinite(float(loss.detach())):
                    raise ValueError("nonfinite TRAIN loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                result["optimizer_updates"] += 1
                if time.monotonic() - started >= 600:
                    result["status"] = "time_cap"
                    break
            if result["status"] == "time_cap":
                break
            tune_loss = _tune_loss(model, tune, started)
            if tune_loss is None:
                result["status"] = "time_cap"
                break
            result["epochs_completed"] = epoch
            result["epoch_history"].append({"epoch": epoch, "tune_nll": tune_loss,
                                            "optimizer_updates": result["optimizer_updates"]})
            if best_loss - tune_loss > 1e-4:
                artifact = bindings | {
                    "schema_version": NEURAL_VERSION, "candidate_key": candidate_key,
                    "architecture": NEURAL_CANDIDATES[candidate_key], "seed": seed,
                    "parameter_count": result["parameter_count"], "input_dimensions": result["input_dimensions"],
                    "tune_ids": list(tune.case_ids), "tune_ids_hash": content_hash(tune.case_ids),
                    "fit_settings": _NEURAL_SETTINGS, "source_hashes": before, "budget": budget,
                    "checkpoint": {"epoch": epoch, "optimizer_updates": result["optimizer_updates"],
                                   "tune_nll": tune_loss},
                    "tensors": {name: value.detach().numpy().tolist() for name, value in model.state_dict().items()},
                    "torch_version": torch.__version__,
                }
                artifact["model_id"] = content_hash(artifact)
                path = _write_json(output_dir / f"checkpoint-{epoch:02d}.json", artifact)
                _write_json(output_dir / "model.json", artifact)
                best_loss, stale_epochs = tune_loss, 0
                result.update({"model_path": str(output_dir / "model.json"), "checkpoint_hash": artifact["model_id"],
                               "checkpoint_file_sha256": _file_hash(output_dir / "model.json"),
                               "best_epoch": epoch, "best_tune_nll": tune_loss})
                result["checkpoints"].append({"path": str(path), "model_id": artifact["model_id"],
                                              "sha256": _file_hash(path), "epoch": epoch, "tune_nll": tune_loss})
            else:
                stale_epochs += 1
            _write_json(output_dir / "progress.json", result | {"elapsed_seconds": time.monotonic() - started})
            if time.monotonic() - started >= 600:
                result["status"] = "time_cap"
                break
            if stale_epochs >= 4:
                result["status"] = "early_stopped"
                break
    except Exception as error:
        result.update({"status": "error", "error_type": type(error).__name__, "error": str(error)})
    result["source_hashes_after"] = source_hashes()
    result["prepared_input_hashes_after"] = prepared_input_hashes(prepared_dir, registration_path)
    if before != result["source_hashes_after"] or inputs_before != result["prepared_input_hashes_after"]:
        result["status"] = "source_changed"
    result["elapsed_seconds"] = time.monotonic() - started
    if result["elapsed_seconds"] >= 600 and result["status"] in {"complete", "early_stopped"}:
        result["status"] = "time_cap"
    result["eligible"] = result["status"] in {"complete", "early_stopped"} and result["model_path"] is not None
    result["result_id"] = content_hash(result)
    _write_json(output_dir / "result.json", result)
    return result
