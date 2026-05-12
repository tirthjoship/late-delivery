"""PyTorch model adapter (stub)."""

from typing import Any


class PyTorchPredictor:
    """Neural network adapter for late delivery risk prediction (stub).

    Implements ModelTrainerPort. Not yet implemented.
    """

    def train(self, X: Any, y: Any) -> None:
        raise NotImplementedError

    def predict(self, X: Any) -> Any:
        raise NotImplementedError

    def predict_proba(self, X: Any) -> Any:
        raise NotImplementedError

    def get_params(self) -> dict[str, object]:
        raise NotImplementedError
