from copy import deepcopy
from typing import Dict, List, Literal

import torch
from torchmetrics.classification.base import _ClassificationTaskWrapper


class MetricsCollection:
    """Collects and computes evaluation metrics and aggregated loss for a model over multiple batches.

    This collector wraps a list of torchmetrics metric objects and accumulates their
    state across calls to ``step``. It also accumulates the scalar loss value so that
    the average or total loss can be reported alongside computed metrics.
    """

    def __init__(self, metrics: Dict[str, _ClassificationTaskWrapper], device: torch.device):
        """Create a MetricsCollection.

        Args:
            metrics: A dictionary mapping metric names to instantiated torchmetrics metric objects.
            device: Device to move metric states and computations to.
        """
        self._device = device
        self._metrics = {name: metric.to(device) for name, metric in deepcopy(metrics).items()}
        self._memory: Dict[str, List[float]] = self._init_memory()
        self._temp_memory: Dict[str, List[float]] = self._init_memory()
        self._steps: int = 0

    def _init_memory(self) -> Dict[str, List[float]]:
        memory = {name: [] for name in self._metrics.keys()}
        memory["loss"] = []
        return memory

    def compute_metrics(self) -> Dict[str, float]:
        """Compute and return all metrics and the accumulated loss.

        Returns:
            A dictionary mapping metric names to scalar values and including the key
            ``"loss"`` for the current accumulated loss.
        """
        result = {}
        for name, steps in self._temp_memory.items():
            if name == "loss":
                result["loss"] = sum(steps) / self._steps
            else:
                final_val = self._metrics[name].compute()
                try:
                    result[name] = float(final_val.item())
                except:
                    result[name] = final_val.tolist()
        return result

    def step(self, outputs: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> Dict[str, float]:
        """Update internal metric state with a new batch and accumulate loss.

        Args:
            outputs: Model outputs for the current batch.
            targets: Ground-truth targets for the current batch.
            loss: Computed loss tensor for the current batch; its scalar value will be accumulated.

        Returns:
            A dictionary mapping metric names to their computed values for the current batch,
            including the key ``"loss"`` for the current batch loss.
        """
        result = {}
        for name, metric in self._metrics.items():
            val: torch.Tensor = metric(outputs, targets)
            try:
                val = float(val.item())
            except:
                val = val.tolist()
            self._temp_memory[name].append(val)
            self._memory[name].append(val)
            result[name] = val

        loss_ = float(loss.item())
        self._temp_memory["loss"].append(loss_)
        self._memory["loss"].append(loss_)
        result["loss"] = loss_

        self._steps += 1
        return result

    def reset(self) -> None:
        """Reset recorded per-step history and cached computations."""
        for metric in self._metrics.values():
            metric.reset()
        self._temp_memory = self._init_memory()
        self._steps = 0

    def get_step_records(self, step: int) -> Dict[str, float]:
        """Return recorded metric/loss values for a specific step index."""
        if step >= self._steps:
            raise IndexError(f"Step index {step} is out of bounds for recorded steps ({-self._steps} to {self._steps - 1})")
        return {name: self._temp_memory[name][step] for name in self._temp_memory.keys()}

    def get_all_saved_records(self) -> Dict[str, List[float]]:
        """Return all persisted recorded metric/loss values across resets."""
        return self._memory

    def format_metrics_to_string(self) -> str:
        """Format computed metrics into a human-readable single-line string for logging.

        The formatting produces ``name: value`` pairs separated by ``|`` and rounds
        numeric values to four decimal places. Lists or tuples are formatted as
        comma-separated numeric entries with the same precision.

        Returns:
            A formatted string suitable for inclusion in logs.
        """
        metrics = self.compute_metrics()
        metrics_log = ""
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if isinstance(value, (list, tuple)):
                metrics_log += f"{key}: " + ", ".join([f"{v}" for v in value]) + " | "
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