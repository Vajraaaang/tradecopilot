"""Deterministic replay controls and explicit recurrent inference-state handling."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class FixedPolicy:
    def __init__(
        self, kind: str, *, feature_index: int = 0, missing_index: int | None = None,
        mean: float = 0, scale: float = 1, threshold: float = 5,
    ):
        if kind not in {"cash", "hold", "rule"}:
            raise ValueError("unknown fixed policy")
        self.kind, self.feature_index = kind, feature_index
        self.mean, self.scale, self.threshold = mean, scale, threshold
        self.missing_index = missing_index

    def reset_memory(self) -> None:
        pass

    def act(self, observation: NDArray[np.float32]) -> int:
        if self.kind == "cash":
            return 0
        if self.kind == "hold":
            return 2
        if self.missing_index is not None and observation[self.missing_index] != 0:
            return 0
        value = float(observation[self.feature_index]) * self.scale + self.mean
        return 2 if value > self.threshold else 0


class NeuralPolicy:
    """Each replay starts fresh; state never crosses a symbol/date or terminal boundary."""

    def __init__(self, model: Any, *, recurrent: bool):
        self.model, self.recurrent = model, recurrent
        self.state: Any = None
        self.start = True

    def reset_memory(self) -> None:
        self.state, self.start = None, True

    def act(self, observation: NDArray[np.float32]) -> int:
        kwargs: dict[str, Any] = {"deterministic": True}
        if self.recurrent:
            kwargs.update(state=self.state, episode_start=np.asarray([self.start], dtype=bool))
        action, next_state = self.model.predict(observation, **kwargs)
        values = np.asarray(action)
        if values.size != 1 or values.dtype.kind not in "iu":
            raise ValueError("policy must return one discrete integer action")
        result = int(values.reshape(-1)[0])
        if result not in (0, 1, 2):
            raise ValueError("policy action outside registered space")
        self.state, self.start = next_state, False
        return result
