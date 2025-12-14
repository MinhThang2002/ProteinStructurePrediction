import os
import time
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from tqdm import tqdm
import sidechainnet as scn
import matplotlib.pyplot as plt

from helper import init_loss_optimizer


# -------------------------
# Utils
# -------------------------
def _get_inputs(batch, mode: str, device: torch.device):
    if mode == "seqs":
        return batch.int_seqs.to(device).long()
    elif mode == "pssms":
        return batch.seq_evo_sec.to(device)
    raise ValueError(f"Unsupported mode: {mode}")


def _get_targets_and_mask(batch, device: torch.device):
    """
    Returns:
      attn_mask: (B, L) mask for attention (batch.msks)
      true_sincos: (B, L, 12, 2) sin/cos targets
      valid_mask: boolean mask (B, L, 12, 2) where angles are not padded (angs != 0)
    """
    attn_mask = batch.msks.to(device)

    angs = batch.angs.to(device)
    angs[~torch.isfinite(angs)] = 0

    true_sincos = scn.structure.trig_transform(angs)
    valid_mask = (angs.ne(0)).unsqueeze(-1).repeat(1, 1, 1, 2)

    return attn_mask, true_sincos, valid_mask


def _rmse_from_diff(diff: torch.Tensor) -> torch.Tensor:
    # diff is 1D tensor after masking
    return torch.sqrt((diff * diff).mean())


# -------------------------
# EMA helper
# -------------------------
class EMA:
    def __init__(self, decay: float):
        self.decay = float(decay)
        self.shadow: Optional[Dict[str, torch.Tensor]] = None

    def enabled(self) -> bool:
        return self.decay > 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        if not self.enabled():
            return
        if self.shadow is None:
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        if self.shadow is None:
            return None
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k, v in model.state_dict().items():
            v.copy_(self.shadow[k])
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: Optional[Dict[str, torch.Tensor]]):
        if backup is None:
            return
        for k, v in model.state_dict().items():
            v.copy_(backup[k])


# -------------------------
# Metric: Standard RMSE (global, element-weighted)
# -------------------------
def validation(model, datasplit, device, loss_fn, mode):
    """
    Standard RMSE over sin/cos targets:
      RMSE = sqrt( mean( (pred - true)^2 ) )
    Computed globally across all valid elements (not per-batch average).
    Note: loss_fn kept for backward compatibility (not used).
    """
    model.eval()
    sum_sq = torch.zeros((), device=device)
    count = 0

    with torch.no_grad():
        for batch in datasplit:
            x = _get_inputs(batch, mode, device)
            attn_mask, true_sincos, valid_mask = _get_targets_and_mask(batch, device)

            if valid_mask.sum() == 0:
                continue

            pred = model(x, mask=attn_mask)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

            diff = pred[valid_mask] - true_sincos[valid_mask]
            sum_sq += (diff * diff).sum()
            count += diff.numel()

    if count == 0:
        return torch.tensor(float("nan"), device=device)

    return torch.sqrt(sum_sq / count)


# -------------------------
# Training
# -------------------------
def train(model, config, dataloader, device):
    steps_per_epoch = len(dataloader["train"])
    accum_steps = max(1, int(config.grad_accum_steps))

    optimizer, scheduler, batch_rmses, epoch_train_rmse, epoch_valid10_rmse, epoch_valid90_rmse, loss_fn = init_loss_optimizer(
        model, config, steps_per_epoch=steps_per_epoch
    )

    os.makedirs(config.model_save_path, exist_ok=True)

    best_metric_file = os.path.join(config.model_save_path, "best_metric.txt")
    run_best_path = os.path.join(config.model_save_path, "model_weights_run_best.pth")
    if os.path.exists(run_best_path):
        os.remove(run_best_path)

    # Load previous best metric (overall)
    prev_best = float("inf")
    if os.path.exists(best_metric_file):
        try:
            with open(best_metric_file, "r") as f:
                prev_best = float(f.read().strip())
        except Exception:
            prev_best = float("inf")

    ema = EMA(getattr(config, "ema_decay", 0.0))
    run_best = float("inf")
    final_save_path = None

    start_time = time.time()

    for epoch in range(config.epoch):
        print(f"Epoch {epoch}")
        model.train()
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(total=steps_per_epoch, smoothing=0)

        for step_idx, batch in enumerate(dataloader["train"]):
            x = _get_inputs(batch, config.mode, device)
            attn_mask, true_sincos, valid_mask = _get_targets_and_mask(batch, device)

            if valid_mask.sum() == 0:
                pbar.update(1)
                pbar.set_description("Batch RMSE = skip")
                continue

            pred = model(x, mask=attn_mask)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

            # Train loss (Huber / SmoothL1)
            loss_raw = loss_fn(pred[valid_mask], true_sincos[valid_mask])
            if not torch.isfinite(loss_raw):
                optimizer.zero_grad(set_to_none=True)
                pbar.update(1)
                pbar.set_description("Batch RMSE = nan-skip")
                continue

            # Log RMSE chuẩn cho batch
            with torch.no_grad():
                diff = pred[valid_mask] - true_sincos[valid_mask]
                rmse_batch = _rmse_from_diff(diff)

            # Backprop (accumulation)
            (loss_raw / accum_steps).backward()

            do_step = ((step_idx + 1) % accum_steps == 0) or ((step_idx + 1) == steps_per_epoch)
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                ema.update(model)

                if scheduler is not None and config.scheduler_step == "batch":
                    scheduler.step()

            batch_rmses.append(float(rmse_batch))
            pbar.update(1)
            pbar.set_description(f"Batch RMSE = {rmse_batch.item():.4f} | Huber = {loss_raw.item():.4f}")

        pbar.close()

        # Evaluate using EMA weights (if enabled)
        backup = ema.apply_to(model) if ema.enabled() else None

        tr = float(validation(model, dataloader["train-eval"], device, loss_fn, config.mode))
        v10 = float(validation(model, dataloader["valid-10"], device, loss_fn, config.mode))
        v90 = float(validation(model, dataloader["valid-90"], device, loss_fn, config.mode))

        epoch_train_rmse.append(tr)
        epoch_valid10_rmse.append(v10)
        epoch_valid90_rmse.append(v90)

        ema.restore(model, backup)

        print(f"     Train-eval RMSE = {tr:.4f}")
        print(f"     Valid-10   RMSE = {v10:.4f}")
        print(f"     Valid-90   RMSE = {v90:.4f}")

        metric_map = {"train-eval": tr, "valid-10": v10, "valid-90": v90}
        current = metric_map[config.best_metric_split]

        if current < run_best:
            run_best = current
            state_to_save = ema.shadow if (ema.enabled() and ema.shadow is not None) else model.state_dict()
            torch.save(state_to_save, run_best_path)
            print(f"     Updated run-best ({config.best_metric_split}) = {run_best:.4f}")

        if scheduler is not None and config.scheduler_step == "epoch":
            scheduler.step()

    # Test (evaluate with EMA if available)
    backup = ema.apply_to(model) if ema.enabled() else None
    test_rmse = float(validation(model, dataloader["test"], device, loss_fn, config.mode))
    ema.restore(model, backup)
    print(f"Test RMSE = {test_rmse:.4f}")

    # Save final checkpoint (overall best logic)
    if run_best < float("inf") and os.path.exists(run_best_path):
        if run_best < prev_best:
            final_save_path = os.path.join(config.model_save_path, "model_weights.pth")
            os.replace(run_best_path, final_save_path)
            with open(best_metric_file, "w") as f:
                f.write(f"{run_best}\n")
            print(f"Saved new overall best to {final_save_path} (metric {run_best:.4f})")
        else:
            fallback = os.path.join(config.model_save_path, f"model_weights_{int(time.time())}.pth")
            os.replace(run_best_path, fallback)
            final_save_path = fallback
            print(f"Run best {run_best:.4f} did not beat previous {prev_best:.4f}; saved to {final_save_path}")
    else:
        fallback = os.path.join(config.model_save_path, f"model_weights_{int(time.time())}.pth")
        torch.save(model.state_dict(), fallback)
        final_save_path = fallback
        print(f"No validation checkpoints saved; saved current model to {final_save_path}")

    # Log training run (append one line)
    log_path = os.path.join(config.model_save_path, "train_log.txt")
    need_header = (not os.path.exists(log_path)) or (os.path.getsize(log_path) == 0)

    header = (
        "timestamp\tepochs\tbatch\tmode\tinteger_seq\t"
        "d_hidden\tdim\td_embedding\theads\thead_dim\tattn_dropout\t"
        "lr\tcurrent_lr\tscheduler\tscheduler_step\twarmup_epochs\tmin_lr\tweight_decay\t"
        "grad_accum_steps\tema_decay\tbest_split\trun_best\ttrain_eval\tvalid10\tvalid90\t"
        "test\tcheckpoint\tduration_sec\n"
    )

    current_lr = optimizer.param_groups[0]["lr"]
    duration_sec = time.time() - start_time

    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t"
        f"{config.epoch}\t{config.batch}\t{config.mode}\t{config.integer_sequence}\t"
        f"{config.d_hidden}\t{config.dim}\t{config.d_embedding}\t{config.n_heads}\t{config.head_dim}\t{config.attn_dropout}\t"
        f"{config.learning_rate}\t{current_lr:.6f}\t{config.scheduler}\t{config.scheduler_step}\t{config.warmup_epochs}\t{config.min_lr}\t{config.weight_decay}\t"
        f"{config.grad_accum_steps}\t{config.ema_decay}\t{config.best_metric_split}\t{run_best:.4f}\t"
        f"{epoch_train_rmse[-1]:.4f}\t{epoch_valid10_rmse[-1]:.4f}\t{epoch_valid90_rmse[-1]:.4f}\t"
        f"{test_rmse:.4f}\t{final_save_path}\t{duration_sec:.1f}\n"
    )

    with open(log_path, "a") as f:
        if need_header:
            f.write(header)
        f.write(line)

    # Plots (saved into model_save_path)
    try:
        train_png = os.path.join(config.model_save_path, "TrainingLoss.png")
        valid_png = os.path.join(config.model_save_path, "ValidationLoss.png")

        plt.figure()
        plt.plot(np.asarray(batch_rmses), label="batch RMSE")
        plt.ylabel("RMSE")
        plt.xlabel("Step")
        plt.title("Training RMSE over Time")
        plt.legend()
        plt.savefig(train_png)
        plt.close()

        plt.figure()
        plt.plot(epoch_train_rmse, label="train-eval")
        plt.plot(epoch_valid10_rmse, label="valid-10")
        plt.plot(epoch_valid90_rmse, label="valid-90")
        plt.ylabel("RMSE")
        plt.xlabel("Epoch")
        plt.title("RMSE on Splits over Time")
        plt.legend()
        plt.savefig(valid_png)
        plt.close()
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    return model, final_save_path