import os
import logging
from tqdm.auto import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler, LinearLR, SequentialLR, ReduceLROnPlateau

from .utils import MetricsCollection, EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """High-level training orchestrator that handles epochs, evaluation and optional early stopping.

    This class encapsulates the model, optimizer and loss function and provides a
    convenience ``train`` method that runs training and validation loops, logs
    progress, and optionally evaluates on a held-out test set.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TENSORBOARD_FOLDER = PROJECT_ROOT / "runs"
    TENSORBOARD_FOLDER.mkdir(parents=True, exist_ok=True)

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        use_amp: bool = False,
        enable_tensorboard: bool = True,
        experiment_name: Optional[str] = None,
        writer: Optional[SummaryWriter] = None,
        scheduler: Optional[_LRScheduler] = None,
        warmup_steps: int = 0,
        step_scheduler_every_batch: bool = False,
    ):
        """Create a Trainer instance.

        Args:
            model: The model to train. It will be moved to ``device``.
            optimizer: Optimizer used for updating model parameters.
            loss_fn: Loss function used to compute gradients during training.
            device: Device where model and data tensors will be placed.
            use_amp: Whether to enable automatic mixed precision (requires CUDA).
            enable_tensorboard: Whether to enable TensorBoard logging. If False, no SummaryWriter will be created.
            experiment_name: Optional name for the TensorBoard experiment. If not provided, a default name will be generated.
            writer: Optional externally-created SummaryWriter instance to use.
            scheduler: Optional ready-to-use PyTorch LR scheduler (any subclass of _LRScheduler).
            warmup_steps: Number of initial steps (batches or epochs depending on stepping mode) to linearly warm up the LR.
            step_scheduler_every_batch: Whether to call scheduler.step() after every optimizer update (per-batch). 
                If False, scheduler.step() is called once per epoch.
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

        if not experiment_name:
            files = os.listdir(self.TENSORBOARD_FOLDER)
            experiment_name = f"experiment_{len(files) + 1}"

        # TensorBoard writer handling
        if enable_tensorboard:
            self.writer = writer if writer is not None else SummaryWriter(str(self.TENSORBOARD_FOLDER / experiment_name))
        else:
            self.writer = None

        # Initialize global step for per-batch TensorBoard logging
        self.global_step = 0

        # Gradient norm tracking
        self.avg_grad_norm = 0.0

        # Scheduler / warmup settings
        self.step_scheduler_every_batch = step_scheduler_every_batch
        self.scheduler = scheduler

        if warmup_steps > 0 and scheduler is not None:
            warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
            self.scheduler = SequentialLR(optimizer, schedulers=[warmup, scheduler], milestones=[warmup_steps])
        
        if self.scheduler is not None and not hasattr(self.scheduler, "step"):
            raise TypeError("scheduler must be a valid PyTorch scheduler with a .step() method")

    def _log_model_graph(self, data_loader: Optional[DataLoader] = None) -> None:
        """Log the model graph to TensorBoard using an example input.

        Args:
            data_loader: DataLoader to draw a single batch from
        """
        if self.writer is None:
            return

        batch = next(iter(data_loader))
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            inp = batch[0]
        else:
            inp = batch

        inp = self._to_device(inp)
        if isinstance(inp, torch.Tensor):
            inp = (inp, )

        try:
            self.model.eval()
            with torch.no_grad():
                class WrappedModel(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    def forward(self, *args):
                        return self.model(*args)

                wrapped = WrappedModel(self.model)

                self.writer.add_graph(wrapped, inp)
                self.writer.flush()
                logger.info("Model graph written to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to write model graph to TensorBoard: {e}")

    def train(
        self, 
        num_epochs: int, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        metrics: Dict[str, Any] = {},
        early_stopping: Optional[Dict[str, Any]] = None,
        log_graph: bool = False,
        is_calc_gradient: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Run the training loop with validation, logging and optional early stopping.

        Args:
            num_epochs: Number of full passes over the training dataset.
            train_loader: DataLoader that yields training batches.
            val_loader: DataLoader used to evaluate validation performance after each epoch.
            test_loader: Optional DataLoader for a final test evaluation after training.
            metrics: Dict of name with torchmetrics metric instances to evaluate during training and validation.
            early_stopping: Optional mapping used to construct an EarlyStopping instance
                (e.g. {"monitor": "loss", "patience": 3}). If omitted, early stopping is disabled.
            log_graph: Optional flag to log the model graph to TensorBoard once before training.
            is_calc_gradient: Optional flag to calculate and print gradient norms during training.

        Returns:
            A dictionary with training, validation and test metrics records.
        """
        train_metrics = MetricsCollection(metrics, self.device)
        val_metrics = MetricsCollection(metrics, self.device)
        stopper = EarlyStopping(**early_stopping) if early_stopping else None

        if log_graph and self.writer is not None:
            self._log_model_graph(data_loader=train_loader)

        for epoch in range(num_epochs):
            self.model.train()
            train_metrics.reset()
            val_metrics.reset()

            self._train(train_loader, train_metrics, is_calc_gradient)
            self._test(val_loader, val_metrics, phase="val")
            self.log(epoch, train_metrics, val_metrics)

            self._step_scheduler_epoch(val_metrics)

            if self.writer is not None:
                self._log_tensorboard_epoch(train_metrics, val_metrics, epoch)

            if stopper:
                if self._handle_early_stopping(stopper, val_metrics):
                    logger.info(f"Early stopping triggered: no improvement in {stopper.monitor}")
                    break

        test_metrics = None
        if test_loader:
            test_metrics = MetricsCollection(metrics, self.device)
            self._test(test_loader, test_metrics, phase="test")
            logger.info(f"Test Metrics: {test_metrics.format_metrics_to_string()}")

            if self.writer is not None:
                test_results = test_metrics.compute_metrics()
                for key, val in test_results.items():
                    if isinstance(val, (int, float)):
                        self.writer.add_scalar(f"test/{key}", val, num_epochs)

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        return {
            "train": self._create_output_dict(train_metrics),
            "val": self._create_output_dict(val_metrics),
            "test": self._create_output_dict(test_metrics) if test_metrics else None,
        }

    def _step_scheduler_epoch(self, val_metrics: MetricsCollection) -> None:
        """Perform scheduler stepping at the end of an epoch if configured.

        This handles both standard schedulers (step once per epoch) and
        ReduceLROnPlateau which requires a monitored metric value from the
        validation metrics.

        Args:
            val_metrics: Validation metrics collection used to obtain the monitor value.

        Returns:
            None
        """
        if self.scheduler is not None and not self.step_scheduler_every_batch:
            try:

                # Handle ReduceLROnPlateau separately
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    val_results = val_metrics.compute_metrics()
                    monitor_value = val_results.get("loss", None)
                    if monitor_value is not None:
                        self.scheduler.step(monitor_value)
                    else:
                        logger.warning("ReduceLROnPlateau requires 'loss' metric; skipping scheduler step.")
                else:
                    self.scheduler.step()
            except Exception as e:
                logger.warning(f"Scheduler step (per-epoch) failed: {e}")

    def _log_tensorboard_epoch(
        self, 
        train_metrics: MetricsCollection, 
        val_metrics: MetricsCollection, 
        epoch: int
    ) -> None:
        """Log training and validation metrics and the learning rate to TensorBoard.

        Args:
            train_metrics: MetricsCollection with accumulated training metrics for the epoch.
            val_metrics: MetricsCollection with accumulated validation metrics for the epoch.
            epoch: Zero-based epoch index used to compute the TensorBoard step.

        Returns:
            None
        """
        train_results = train_metrics.compute_metrics()
        val_results = val_metrics.compute_metrics()
        step = epoch + 1
        for key, val in train_results.items():
            if isinstance(val, (int, float)):
                self.writer.add_scalar(f"train/{key}", val, step)
        for key, val in val_results.items():
            if isinstance(val, (int, float)):
                self.writer.add_scalar(f"val/{key}", val, step)

        # log LR
        try:
            lr = self.optimizer.param_groups[0].get("lr", None)
            if lr is not None:
                self.writer.add_scalar("lr/epoch", lr, step)
        except Exception as e:
            logger.warning(f"Failed to log LR: {e}")

    def _handle_early_stopping(self, stopper: EarlyStopping, val_metrics: MetricsCollection) -> bool:
        """Evaluate the early stopping criterion using validation metrics.

        Args:
            stopper: EarlyStopping instance tracking the monitored metric and patience.
            val_metrics: Validation metrics collection from which the monitored value is read.

        Returns:
            True if training should be stopped according to the stopper, otherwise False.
        """
        val_results = val_metrics.compute_metrics()
        monitor_value = val_results.get(stopper.monitor)
        if monitor_value is not None:
            try:
                monitor_value = float(monitor_value)
                return stopper.step(monitor_value)
            except (TypeError, ValueError):
                logger.warning(f"Invalid monitor value for early stopping: {monitor_value}")
        return False

    def _create_output_dict(self, mc: MetricsCollection) -> Dict[str, Any]:
        """Build the serialized output dictionary from a MetricsCollection.

        The returned mapping contains each metric name mapping to a dict with
        recorded stepwise values under "steps" and the final aggregated
        value under "final".

        Args:
            mc: MetricsCollection to serialize.

        Returns:
            A dictionary mapping metric names to their recorded steps and final value.
        """
        result = {}
        metrics = mc.compute_metrics()
        for name, vals in mc.get_all_saved_records().items():
            result[name] = {"steps": vals, "final": metrics[name]}
        return result

    def _train(self, train_loader: DataLoader, metrics: MetricsCollection, is_calc_gradients: bool) -> None:
        """Perform a single training epoch over the provided DataLoader.

        Args:
            train_loader: DataLoader yielding training batches.
            metrics_eval: MetricsEvaluator instance used to update and accumulate metrics.
            is_calc_gradients: Optional flag to calculate and print gradient norms during training.
        """
        acc_h_arr = []
        total_norm = 0.0
        batch_grad_norm = 0.0
        num_batches = 0
        self.model.train()
        pbar = tqdm(train_loader, desc="Train", unit="batch", total=len(train_loader))
        for batch in pbar:
            x_batch = self._to_device(batch[0])
            y_batch = self._to_device(batch[1])

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(device_type='cuda'):
                    if isinstance(x_batch, list):
                        outputs = self.model(*x_batch)
                    else:
                        outputs = self.model(x_batch)
                    loss = self.loss_fn(outputs, y_batch)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                if is_calc_gradients:
                    batch_grad_norm = self._calc_gradients()
                    total_norm += batch_grad_norm
                    num_batches += 1

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()

                metrics.step(self._ordinal_to_class(outputs), y_batch, loss)

                # scheduler step per-batch (if configured)
                if self.scheduler is not None and self.step_scheduler_every_batch:
                    self.scheduler.step()
            else:
                if isinstance(x_batch, list):
                    outputs = self.model(*x_batch)
                else:
                    outputs = self.model(x_batch)
                loss = self.loss_fn(outputs, y_batch)

                loss.backward()

                if is_calc_gradients:
                    batch_grad_norm = self._calc_gradients()
                    total_norm += batch_grad_norm
                    num_batches += 1

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                metrics.step(self._ordinal_to_class(outputs), y_batch, loss)

                # scheduler step per-batch (if configured)
                if self.scheduler is not None and self.step_scheduler_every_batch:
                    self.scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "grad_norm": f"{batch_grad_norm:.4f}"})

            B, C, K = outputs.shape
            batch_acc_h = []
            for k in range(K):
                preds_k = self._ordinal_to_class(outputs[:, :, k])      # (B,)
                target_k = y_batch[:, k]                           # (B,)
                acc_k = (preds_k == target_k).float().mean()       # scalar
                batch_acc_h.append(acc_k)
            acc_h_arr.append(torch.stack(batch_acc_h))  # (K,)

            if self.writer is not None:
                try:
                    loss_val = float(loss.item())
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    lr = self.optimizer.param_groups[0].get("lr", None)
                    if lr is not None:
                        self.writer.add_scalar("lr/batch", lr, self.global_step)
                except Exception as e:
                    logger.warning(f"Failed to log batch scalars to TensorBoard: {e}")
                finally:
                    self.global_step += 1

        acc_h = torch.stack(acc_h_arr, dim=0).mean(dim=0)  # (K,)
        print(*acc_h.tolist(), float(acc_h.mean().item()))
        self.avg_grad_norm = total_norm / num_batches if num_batches > 0 else 0.0

    def _calc_gradients(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def _test(self, test_loader: DataLoader, metrics: MetricsCollection, phase: str = "val") -> None:
        """Evaluate the model on data from ``test_loader`` without updating parameters.

        Args:
            test_loader: DataLoader yielding evaluation batches.
            metrics_eval: MetricsEvaluator used to accumulate metrics for the evaluation set.
            phase: Name of the phase used for TensorBoard logging (e.g. 'val' or 'test').
        """
        acc_h_arr = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test", unit="batch", total=len(test_loader)):
                x_batch = self._to_device(batch[0])
                y_batch = self._to_device(batch[1])
                if self.use_amp:
                    with autocast(device_type="cuda"):
                        if isinstance(x_batch, list):
                            outputs = self.model(*x_batch)
                        else:
                            outputs = self.model(x_batch)
                        loss = self.loss_fn(outputs, y_batch)
                else:
                    if isinstance(x_batch, list):
                        outputs = self.model(*x_batch)
                    else:
                        outputs = self.model(x_batch)
                    loss = self.loss_fn(outputs, y_batch)
                metrics.step(self._ordinal_to_class(outputs), y_batch, loss)

                B, C, K = outputs.shape
                batch_acc_h = []
                for k in range(K):
                    preds_k = self._ordinal_to_class(outputs[:, :, k])      # (B,)
                    target_k = y_batch[:, k]                           # (B,)
                    acc_k = (preds_k == target_k).float().mean()       # scalar
                    batch_acc_h.append(acc_k)
                acc_h_arr.append(torch.stack(batch_acc_h))  # (K,)

                if self.writer is not None:
                    try:
                        loss_val = float(loss.item())
                        self.writer.add_scalar(f"{phase}/loss", loss_val, self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log batch scalars to TensorBoard: {e}")
                    finally:
                        self.global_step += 1

            acc_h = torch.stack(acc_h_arr, dim=0).mean(dim=0)  # (K,)
            print(*acc_h.tolist(), float(acc_h.mean().item()))

    def _to_device(
        self, 
        batch: Union[List[torch.Tensor], torch.Tensor]
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if isinstance(batch, list):
            return [x.to(self.device, non_blocking=True) for x in batch]
        return batch.to(self.device, non_blocking=True)

    def log(
        self, 
        epoch: int, 
        train_metrics_eval: MetricsCollection,
        val_metrics_eval: MetricsCollection
    ) -> None:
        """Log training and validation metrics for the given epoch.

        Args:
            epoch: Zero-based epoch index that was just completed.
            train_metrics_eval: MetricsEvaluator with accumulated training metrics.
            val_metrics_eval: MetricsEvaluator with accumulated validation metrics.
        """
        log_msg = f"Epoch {epoch+1}:\n"
        if self.avg_grad_norm > 0.0:
            log_msg += f"Avg Grad Norm: {self.avg_grad_norm:.4f}\n"
        log_msg += f"Train Metrics: {train_metrics_eval.format_metrics_to_string()}\n"
        log_msg += f"Val Metrics:   {val_metrics_eval.format_metrics_to_string()}\n"
        logger.info(log_msg)

    def _ordinal_to_class(self, logits: torch.Tensor, thresh: float = 0.5) -> torch.Tensor:
        """Convert ordinal logits to class predictions

        The conversion applies a sigmoid activation to the logits and
        counts the number of thresholds exceeded to determine the class.
        """
        # logits: [B, 4, K] -> classes [B, K] through sigmoids
        return (torch.sigmoid(logits) > thresh).sum(dim=1).long()