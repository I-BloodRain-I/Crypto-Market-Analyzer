"""Training utilities for model fitting, evaluation and early stopping.

This module provides lightweight helpers used during model training:

- MetricsEvaluator: Wraps torchmetrics metric objects and accumulates metric
  state and loss across multiple batches.
- EarlyStopping: Monitors a single metric and signals when training should stop
  after a configurable patience without improvement.
- Trainer: High-level orchestrator that runs training and validation loops,
  logs metrics and supports optional early stopping and final test evaluation.

Example:
    trainer = Trainer(model, optimizer, loss_fn, device)
    trainer.train(num_epochs, train_loader, val_loader, metrics=[...])
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from torchmetrics import Accuracy, Precision, Recall, F1Score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """Collects and computes evaluation metrics and aggregated loss for a model over multiple batches.

    This evaluator wraps a list of torchmetrics metric objects and accumulates their
    state across calls to ``step``. It also accumulates the scalar loss value so that
    the average or total loss can be reported alongside computed metrics.
    """

    def __init__(self, metrics: List[F1Score], device: torch.device):
        """Create a MetricsEvaluator.

        Args:
            metrics: Iterable of instantiated torchmetrics metric objects.
            device: Device to move metric states and computations to.
        """
        self._device = device
        self._metrics = {metric.__class__.__name__: metric.to(device) for metric in deepcopy(metrics)}
        self._loss = 0.0
        self._cache = {}
        self._computed = False

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute and return all metrics and the accumulated loss.

        The computed results are cached so repeated calls return the cached values
        until a state-changing method (for example, ``step`` or ``reset``) is invoked.

        Returns:
            A dictionary mapping metric names to scalar values and including the key
            ``"loss"`` for the current accumulated loss.
        """
        if self._computed:
            return self._cache
        results = {name: metric.compute().item() for name, metric in self._metrics.items()}
        results["loss"] = self._loss
        self._computed = True
        self._cache = deepcopy(results)
        return results

    def step(self, outputs: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor):
        """Update internal metric state with a new batch and accumulate loss.

        Args:
            outputs: Model outputs for the current batch.
            targets: Ground-truth targets for the current batch.
            loss: Computed loss tensor for the current batch; its scalar value will be accumulated.
        """
        for metric in self._metrics.values():
            metric.update(outputs, targets)
        self._loss += loss.item()
        self._computed = False
        self._cache.clear()
    
    def reset(self):
        """Reset all metric states and the accumulated loss.

        After calling this method the evaluator behaves as if no batches have been
        seen; computed results will be recomputed on the next call to
        ``compute_metrics``.
        """
        for metric in self._metrics.values():
            metric.reset()
        self._loss = 0.0
        self._computed = False
        self._cache.clear()

    def format_metrics_to_log(self) -> str:
        """Format computed metrics into a human-readable single-line string for logging.

        The formatting produces ``name: value`` pairs separated by ``|`` and rounds
        numeric values to four decimal places. Lists or tuples are formatted as
        comma-separated numeric entries with the same precision.

        Returns:
            A formatted string suitable for inclusion in logs.
        """
        metrics = self.compute_metrics()
        metrics_log = ""
        for key, value in metrics.items():
            if isinstance(value, (list, tuple)):
                metrics_log += f"{key}: " + ", ".join([f"{v:.4f}" for v in value]) + " | "
            else:
                metrics_log += f"{key}: {value:.4f} | "
        return metrics_log[:-3]  # Remove last ' | '


class EarlyStopping:
    """Simple early stopping helper that monitors a single metric and signals when training should stop.

    The monitor name is expected to match a key returned by a MetricsEvaluator's
    ``compute_metrics`` result. The stopping decision is based on the absence of
    improvement for ``patience`` consecutive checks.
    """

    def __init__(
        self,
        monitor: str = 'loss',
        patience: int = 3,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0
    ):
        """Initialize an EarlyStopping instance.

        Args:
            monitor: Name of the metric to watch (must match keys from MetricsEvaluator).
            patience: Number of consecutive non-improving checks before signaling stop.
            mode: Whether lower ("min") or higher ("max") values indicate improvement.
            min_delta: Minimum change in the monitored metric to qualify as an improvement.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float('inf') if self.mode == 'min' else -float('inf')
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        """Advance the early stopping state with the latest monitored value.

        Args:
            value: Latest value of the monitored metric.

        Returns:
            True if the number of consecutive non-improving checks has reached
            or exceeded ``patience`` (i.e. training should stop), otherwise False.
        """
        if not isinstance(value, float):
            raise TypeError(f"Monitored metric '{self.monitor}' should be a float, got {type(value)}")
        
        if self.mode == 'min':
            improved = value < (self.best - self.min_delta)
        else:
            improved = value > (self.best + self.min_delta)

        if improved:
            self.best = value
            self.num_bad_epochs = 0
            return False
        else:
            self.num_bad_epochs += 1
            return self.num_bad_epochs >= self.patience


class Trainer:
    """High-level training orchestrator that handles epochs, evaluation and optional early stopping.

    This class encapsulates the model, optimizer and loss function and provides a
    convenience ``train`` method that runs training and validation loops, logs
    progress, and optionally evaluates on a held-out test set.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        use_amp: bool = False,
        log_dir: Optional[str] = None,
        writer: Optional[SummaryWriter] = None,
    ):
        """Create a Trainer instance.

        Args:
            model: The model to train. It will be moved to ``device``.
            optimizer: Optimizer used for updating model parameters.
            loss_fn: Loss function used to compute gradients during training.
            device: Device where model and data tensors will be placed.
            use_amp: Whether to enable automatic mixed precision (requires CUDA).
            log_dir: Optional path to a TensorBoard log directory. If provided and
                ``writer`` is not supplied, a SummaryWriter will be created.
            writer: Optional externally-created SummaryWriter instance to use.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        if use_amp and not torch.cuda.is_available():
            logger.warning("CUDA not available, disabling AMP (use_amp=False)")
            use_amp = False
        if use_amp and getattr(device, 'type', None) != 'cuda':
            logger.warning("Device is not CUDA, disabling AMP (use_amp=False)")
            use_amp = False

        self.use_amp = use_amp
        self.scaler = GradScaler() if self.use_amp else None

        # TensorBoard writer handling
        self.writer = writer if writer is not None else (SummaryWriter(log_dir) if log_dir is not None else None)
        self._owns_writer = (writer is None) and (log_dir is not None)

    def _log_model_graph(self, data_loader: Optional[DataLoader] = None, example_input: Optional[Any] = None) -> None:
        """Log the model graph to TensorBoard using an example input.

        The method attempts to use ``example_input`` if provided; otherwise it will
        try to pull a single batch from ``data_loader``. Handles single-tensor
        inputs as well as list/tuple inputs (for models accepting multiple inputs).

        Args:
            data_loader: DataLoader to draw a single batch from when ``example_input`` is not provided.
            example_input: Optional example input compatible with the model. Can be a
                tensor, tuple, list or other object the model accepts.
        """
        if self.writer is None:
            return

        inp = example_input
        if inp is None and data_loader is not None:
            try:
                batch = next(iter(data_loader))
            except Exception as e:
                logger.warning(f"Unable to fetch a batch from DataLoader to log model graph: {e}")
                return
            # Expecting (x_batch, y_batch) tuples from the DataLoader
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                inp = batch[0]
            else:
                inp = batch

        if inp is None:
            logger.warning("No example input available to log model graph")
            return

        try:
            if isinstance(inp, (list, tuple)):
                prepared = []
                for x in inp:
                    if isinstance(x, torch.Tensor):
                        prepared.append(x.to(self.device))
                    else:
                        prepared.append(x)
                prepared_inp = tuple(prepared)
            elif isinstance(inp, torch.Tensor):
                prepared_inp = inp.to(self.device)
            else:
                prepared_inp = inp

            try:
                self.writer.add_graph(self.model, prepared_inp)
                self.writer.flush()
                logger.info("Model graph written to TensorBoard")
            except Exception as e:
                logger.warning(f"Failed to write model graph to TensorBoard: {e}")
        except Exception as e:
            logger.warning(f"Error preparing example input for model graph: {e}")

    def train(
        self, 
        num_epochs: int, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        metrics: List[Any] = [],
        early_stopping: Optional[Dict[str, Any]] = None,
        log_graph: bool = False,
        example_input: Optional[Any] = None,
    ):
        """Run the training loop with validation, logging and optional early stopping.

        Args:
            num_epochs: Number of full passes over the training dataset.
            train_loader: DataLoader that yields training batches.
            val_loader: DataLoader used to evaluate validation performance after each epoch.
            test_loader: Optional DataLoader for a final test evaluation after training.
            metrics: List of torchmetrics metric instances to evaluate during training and validation.
            early_stopping: Optional mapping used to construct an EarlyStopping instance
                (e.g. {"monitor": "loss", "patience": 3}). If omitted, early stopping is disabled.
            log_graph: Optional flag to log the model graph to TensorBoard once before training.
            example_input: Optional example input for logging the model graph. Overrides
                ``log_graph`` if provided.
        """
        train_metrics_eval = MetricsEvaluator(metrics, self.device)
        val_metrics_eval = MetricsEvaluator(metrics, self.device)

        stopper = EarlyStopping(**early_stopping) if early_stopping else None

        # Optionally log the model graph once before training
        if log_graph and self.writer is not None:
            self._log_model_graph(data_loader=train_loader, example_input=example_input)
        
        for epoch in range(num_epochs):
            self._train(train_loader, train_metrics_eval)
            self._test(val_loader, val_metrics_eval)
            self.log(epoch, train_metrics_eval, val_metrics_eval)

            # Write metrics to TensorBoard
            if self.writer is not None:
                train_results = train_metrics_eval.compute_metrics()
                val_results = val_metrics_eval.compute_metrics()
                step = epoch + 1
                for key, val in train_results.items():
                    self.writer.add_scalar(f"train/{key}", float(val), step)
                for key, val in val_results.items():
                    self.writer.add_scalar(f"val/{key}", float(val), step)

            # Early stopping check
            if stopper:
                val_results = val_metrics_eval.compute_metrics()
                monitor_value = val_results.get(stopper.monitor)
                if monitor_value is None:
                    logger.warning(f"EarlyStopping monitor '{stopper.monitor}' not found in validation results; skipping early-stopping check")
                else:
                    try:
                        monitor_value = float(monitor_value)
                    except (TypeError, ValueError):
                        logger.warning(f"EarlyStopping monitor '{stopper.monitor}' value is not a float: {monitor_value}; skipping early-stopping check")
                    else:
                        if stopper.step(monitor_value):
                            logger.info(f"Early stopping triggered (no improvement in '{stopper.monitor}' for {stopper.patience} epochs)")
                            break

            train_metrics_eval.reset()
            val_metrics_eval.reset()

        if test_loader:
            test_metrics_eval = MetricsEvaluator(metrics, self.device)
            self._test(test_loader, test_metrics_eval)
            logger.info(f"Test Metrics: {test_metrics_eval.format_metrics_to_log()}")

            if self.writer is not None:
                test_results = test_metrics_eval.compute_metrics()
                step = num_epochs
                for key, val in test_results.items():
                    self.writer.add_scalar(f"test/{key}", float(val), step)

        if self.writer is not None and self._owns_writer:
            self.writer.flush()
            self.writer.close()

    def _train(self, train_loader: DataLoader, metrics_eval: MetricsEvaluator) -> None:
        """Perform a single training epoch over the provided DataLoader.

        Args:
            train_loader: DataLoader yielding training batches.
            metrics_eval: MetricsEvaluator instance used to update and accumulate metrics.
        """
        self.model.train()
        for x_batch, y_batch in train_loader:
            if isinstance(x_batch, (list, tuple)):
                x_batch = [x.to(self.device, non_blocking=True) for x in x_batch]
            else:
                x_batch = x_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(device_type='cuda'):
                    if isinstance(x_batch, list):
                        outputs = self.model(*x_batch)
                    else:
                        outputs = self.model(x_batch)
                    loss = self.loss_fn(outputs.squeeze(-1), y_batch)
                self.scaler.scale(loss).backward()
                metrics_eval.step(outputs.squeeze(-1), y_batch, loss)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if isinstance(x_batch, list):
                    outputs = self.model(*x_batch)
                else:
                    outputs = self.model(x_batch)
                loss = self.loss_fn(outputs.squeeze(-1), y_batch)
                loss.backward()
                metrics_eval.step(outputs.squeeze(-1), y_batch, loss)
                self.optimizer.step()

    def _test(self, test_loader: DataLoader, metrics_eval: MetricsEvaluator) -> None:
        """Evaluate the model on data from ``test_loader`` without updating parameters.

        Args:
            test_loader: DataLoader yielding evaluation batches.
            metrics_eval: MetricsEvaluator used to accumulate metrics for the evaluation set.
        """
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                if self.use_amp:
                    with autocast():
                        if isinstance(x_batch, list):
                            outputs = self.model(*x_batch)
                        else:
                            outputs = self.model(x_batch)
                        loss = self.loss_fn(outputs.squeeze(-1), y_batch)
                else:
                    if isinstance(x_batch, list):
                        outputs = self.model(*x_batch)
                    else:
                        outputs = self.model(x_batch)
                    loss = self.loss_fn(outputs.squeeze(-1), y_batch)
                metrics_eval.step(outputs, y_batch, loss)

    def log(
        self, 
        epoch: int, 
        train_metrics_eval: MetricsEvaluator,
        val_metrics_eval: MetricsEvaluator
    ) -> None:
        """Log training and validation metrics for the given epoch.

        Args:
            epoch: Zero-based epoch index that was just completed.
            train_metrics_eval: MetricsEvaluator with accumulated training metrics.
            val_metrics_eval: MetricsEvaluator with accumulated validation metrics.
        """
        log_msg = f"Epoch {epoch+1}:\n"
        log_msg += f"Train Metrics: {train_metrics_eval.format_metrics_to_log()}\n"
        log_msg += f"Val Metrics:   {val_metrics_eval.format_metrics_to_log()}\n"
        logger.info(log_msg)


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics = MetricsEvaluator(
        metrics=[
            Accuracy(task="multiclass", num_classes=2), 
            Precision(task="multiclass", num_classes=2), 
            Recall(task="multiclass", num_classes=2), 
            F1Score(task="multiclass", num_classes=2)
        ],
        device=device
    )

    trainer = Trainer(model, optimizer, loss_fn, device, use_amp=True)

    # Dummy DataLoader for illustration; replace with actual data loaders
    train_loader = DataLoader(
        [(
            torch.randn(10), 
            torch.randint(0, 2, (1,)).item()
        ) for _ in range(100)], 
        batch_size=32
    )
    val_loader = DataLoader(
        [(
            torch.randn(10), 
            torch.randint(0, 2, (1,)).item()
        ) for _ in range(20)], 
        batch_size=32
    )

    trainer.train(
        num_epochs=50, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        metrics=[
            Accuracy(task="multiclass", num_classes=2), 
            Precision(task="multiclass", num_classes=2), 
            Recall(task="multiclass", num_classes=2), 
            F1Score(task="multiclass", num_classes=2)
        ],
        early_stopping={"monitor": "MulticlassAccuracy", "patience": 5, "mode": "max"}
    )