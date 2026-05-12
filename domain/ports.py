"""Abstract interfaces (protocols) for Retail Supply Chain.

Adapters implement these ports; domain and application depend only
on these abstractions.
"""

from typing import Any, Protocol

from .models import Order, Product


class SalesDataRepository(Protocol):
    """Port: load historical sales/order data.

    Implemented by adapters (e.g. CSV, database) to provide orders
    and optionally products without the domain depending on I/O.
    """

    def get_orders(self) -> list[Order]:
        """Return all orders (and their line items) from the source.

        Returns:
            List of domain Order entities. Order items are grouped
            by Order Id. Leakage columns (e.g. real shipping days,
            delivery status) must not be used.
        """
        ...

    def get_products(self) -> list[Product]:
        """Return distinct products from the source.

        Returns:
            List of domain Product entities.
        """
        ...


class ModelTrainerPort(Protocol):
    """Port: train and predict late delivery risk."""

    def train(self, X: Any, y: Any) -> None:
        """Fit the model on features X and labels y."""
        ...

    def predict(self, X: Any) -> Any:
        """Return binary class predictions for X."""
        ...

    def predict_proba(self, X: Any) -> Any:
        """Return class probability estimates for X."""
        ...


class ExplainerPort(Protocol):
    """Port: explain model predictions."""

    def explain_global(self, X: Any) -> Any:
        """Return global feature importance explanations over dataset X."""
        ...

    def explain_local(self, X: Any, index: int) -> Any:
        """Return a local explanation for the sample at position index in X."""
        ...


class ExperimentTrackerPort(Protocol):
    """Port: track experiments, log metrics, register models."""

    def start_run(self, run_name: str) -> None:
        """Begin a new experiment run with the given name."""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters for the active run."""
        ...

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log a dictionary of evaluation metrics for the active run."""
        ...

    def log_model(self, model: Any, artifact_path: str) -> None:
        """Persist a trained model artifact at the given path."""
        ...

    def log_artifact(self, artifact: Any, artifact_path: str) -> None:
        """Persist an arbitrary artifact (e.g. chart, report) at the given path."""
        ...

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        """Load a registered model by name and lifecycle stage."""
        ...

    def load_artifact(self, run_id: str, artifact_path: str) -> Any:
        """Load an artifact from a specific run by its path."""
        ...

    def register_model(self, model_name: str) -> None:
        """Register the most recently logged model under the given name."""
        ...

    def end_run(self) -> None:
        """Close the active experiment run."""
        ...
