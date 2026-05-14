"""PyTorch neural network adapter (intentionally unused).

XGBoost outperforms neural networks on structured tabular data with 12 features.
This stub demonstrates hexagonal extensibility — any sklearn-compatible model
can plug into ModelTrainerPort without touching domain logic.
See SHAP analysis in notebooks/train_pipeline.ipynb for model comparison evidence.
"""

from typing import Any


class PyTorchPredictor:
    """Neural network adapter stub. See module docstring for rationale."""

    def train(self, X: Any, y: Any) -> None:
        raise NotImplementedError(
            "PyTorch adapter intentionally unimplemented — see module docstring"
        )

    def predict(self, X: Any) -> Any:
        raise NotImplementedError(
            "PyTorch adapter intentionally unimplemented — see module docstring"
        )

    def predict_proba(self, X: Any) -> Any:
        raise NotImplementedError(
            "PyTorch adapter intentionally unimplemented — see module docstring"
        )

    def get_params(self) -> dict[str, object]:
        raise NotImplementedError(
            "PyTorch adapter intentionally unimplemented — see module docstring"
        )
