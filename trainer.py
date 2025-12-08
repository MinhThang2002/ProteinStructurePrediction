import os
import time
import torch
from tqdm import tqdm
import numpy as np
import sidechainnet as scn
from helper import init_loss_optimizer
import matplotlib.pyplot as plt

def validation(model, datasplit, device, loss_fn, mode):
    """Evaluate a model (sequence->sin/cos represented angles [-1,1]) on RMSE over sin/cos."""
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in datasplit:
            # Prepare variables and create a mask of missing angles (padded with zeros)
            # The mask is repeated in the last dimension to match the sin/cos represenation.
            if mode == 'seqs':
                seqs = batch.int_seqs.to(device).long()
            elif mode == 'pssms':
                seqs = batch.seq_evo_sec.to(device)
            mask_ = batch.msks.to(device)
            angs = batch.angs.to(device)
            angs[~torch.isfinite(angs)] = 0
            true_angles_sincosine = scn.structure.trig_transform(angs)
            mask = (angs.ne(0)).unsqueeze(-1).repeat(1, 1, 1, 2)

            if mask.sum() == 0:
                # Skip batches with no valid angle targets.
                continue

            # Make predictions and optimize
            predicted_angles = model(seqs, mask = mask_)
            predicted_angles = torch.nan_to_num(predicted_angles, nan=0.0, posinf=0.0, neginf=0.0)
            loss = loss_fn(predicted_angles[mask], true_angles_sincosine[mask])
            if not torch.isfinite(loss):
                continue
            
            total += loss
            n += 1

    return torch.sqrt(total/n)

def train(model, config, dataloader, device):
    steps_per_epoch = len(dataloader['train'])
    accum_steps = max(1, int(config.grad_accum_steps))
    optimizer, scheduler, batch_losses, epoch_training_losses, epoch_validation10_losses, epoch_validation90_losses, mse_loss = init_loss_optimizer(model, config, steps_per_epoch=steps_per_epoch)
    os.makedirs(config.model_save_path, exist_ok=True)
    best_metric_file = os.path.join(config.model_save_path, 'best_metric.txt')
    start_time = time.time()
    prev_best_metric = float('inf')
    if os.path.exists(best_metric_file):
        try:
            with open(best_metric_file, 'r') as f:
                prev_best_metric = float(f.read().strip())
        except ValueError:
            prev_best_metric = float('inf')
    run_best_metric = float('inf')
    run_best_path = os.path.join(config.model_save_path, 'model_weights_run_best.pth')
    if os.path.exists(run_best_path):
        os.remove(run_best_path)
    final_save_path = None
    ema_state = None

    def update_ema():
        nonlocal ema_state
        if config.ema_decay <= 0:
            return
        with torch.no_grad():
            if ema_state is None:
                ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                for k, v in model.state_dict().items():
                    ema_state[k].mul_(config.ema_decay).add_(v.detach(), alpha=1.0 - config.ema_decay)

    def load_weights_for_eval():
        if ema_state is None:
            return None
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                v.copy_(ema_state[k])
        return backup

    def restore_weights(backup):
        if backup is None:
            return
        with torch.no_grad():
            for k, v in model.state_dict().items():
                v.copy_(backup[k])

    for epoch in range(config.epoch):
        print(f'Epoch {epoch}')
        progress_bar = tqdm(total=len(dataloader['train']), smoothing=0)
        optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(dataloader['train']):
            # Prepare variables and create a mask of missing angles (padded with zeros)
            # Note the mask is repeated in the last dimension to match the sin/cos represenation.
            if config.mode == 'seqs':
                seqs = batch.int_seqs.to(device).long()
            elif config.mode == 'pssms':
                seqs = batch.seq_evo_sec.to(device)
            mask_ = batch.msks.to(device)
            angs = batch.angs.to(device)
            angs[~torch.isfinite(angs)] = 0
            true_angles_sincos = scn.structure.trig_transform(angs)
            mask = (angs.ne(0)).unsqueeze(-1).repeat(1, 1, 1, 2)

            if mask.sum() == 0:
                progress_bar.update(1)
                progress_bar.set_description("\rRMSE Loss = skip")
                continue

            # Make predictions and optimize
            predicted_angles = model(seqs, mask = mask_)
            predicted_angles = torch.nan_to_num(predicted_angles, nan=0.0, posinf=0.0, neginf=0.0)
            loss = mse_loss(predicted_angles[mask], true_angles_sincos[mask])
            if not torch.isfinite(loss):
                progress_bar.update(1)
                progress_bar.set_description("\rRMSE Loss = nan-skip")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss = loss / accum_steps
            loss.backward()

            do_step = ((step_idx + 1) % accum_steps == 0) or ((step_idx + 1) == steps_per_epoch)
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                update_ema()
                if scheduler is not None and config.scheduler_step == 'batch':
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Housekeeping
            batch_losses.append(float(loss))
            progress_bar.update(1)
            progress_bar.set_description(f"\rRMSE Loss = {np.sqrt(float(loss)):.4f}")
        # Evaluate with EMA weights if available
        backup_weights = load_weights_for_eval()
        epoch_training_losses.append(validation(model,
                                                dataloader['train-eval'],
                                                device,
                                                mse_loss,
                                                config.mode))
        # Evaluate the model's performance on various validation sets
        epoch_validation10_losses.append(validation(model,
                                        dataloader['valid-10'],
                                        device,
                                        mse_loss,
                                        config.mode))
        epoch_validation90_losses.append(validation(model,
                                        dataloader['valid-90'],
                                        device,
                                        mse_loss,
                                        config.mode))
        restore_weights(backup_weights)
        print(f"     Train-eval loss = {epoch_training_losses[-1]:.4f}")
        print(f"     Valid-10   loss = {epoch_validation10_losses[-1]:.4f}")
        print(f"     Valid-90   loss = {epoch_validation90_losses[-1]:.4f}")
        metric_map = {
            'train-eval': epoch_training_losses[-1],
            'valid-10': epoch_validation10_losses[-1],
            'valid-90': epoch_validation90_losses[-1],
        }
        current_metric = metric_map[config.best_metric_split]
        if current_metric < run_best_metric:
            run_best_metric = current_metric
            # save EMA weights if available, else raw
            state_to_save = ema_state if ema_state is not None else model.state_dict()
            torch.save(state_to_save, run_best_path)
            print(f"     Updated run-best ({config.best_metric_split}) = {run_best_metric:.4f}")
        if scheduler is not None and config.scheduler_step == 'epoch':
            scheduler.step()
    # Finally, evaluate the model on the test set
    backup_weights = load_weights_for_eval()
    test_loss = validation(model, dataloader['test'], device, mse_loss, config.mode)
    restore_weights(backup_weights)
    test_loss_val = float(test_loss)
    print(f"Test loss = {test_loss_val:.4f}")
    # Decide where to store the run's best checkpoint
    if run_best_metric < float('inf') and os.path.exists(run_best_path):
        if run_best_metric < prev_best_metric:
            final_save_path = os.path.join(config.model_save_path, 'model_weights.pth')
            os.replace(run_best_path, final_save_path)
            with open(best_metric_file, 'w') as f:
                f.write(str(run_best_metric))
            print(f"Saved new overall best to {final_save_path} (metric {run_best_metric:.4f})")
        else:
            fallback_name = f"model_weights_{int(time.time())}.pth"
            final_save_path = os.path.join(config.model_save_path, fallback_name)
            os.replace(run_best_path, final_save_path)
            print(f"Run best {run_best_metric:.4f} did not beat previous {prev_best_metric:.4f}; saved to {final_save_path}")
    else:
        fallback_name = f"model_weights_{int(time.time())}.pth"
        final_save_path = os.path.join(config.model_save_path, fallback_name)
        torch.save(model.state_dict(), final_save_path)
        print(f"No validation checkpoints saved; saved current model to {final_save_path}")
    # Log summary to a text file for reporting
    def _fmt(val):
        try:
            if torch.is_tensor(val):
                val = val.item()
            return f"{float(val):.4f}"
        except Exception:
            return "nan"

    train_eval_last = epoch_training_losses[-1] if epoch_training_losses else float('nan')
    valid10_last = epoch_validation10_losses[-1] if epoch_validation10_losses else float('nan')
    valid90_last = epoch_validation90_losses[-1] if epoch_validation90_losses else float('nan')
    current_lr = optimizer.param_groups[0]['lr']
    duration_sec = time.time() - start_time

    log_path = os.path.join(config.model_save_path, 'train_log.txt')
    header = (
        "timestamp\tepochs\tbatch\tmode\tinteger_seq\t"
        "d_hidden\tdim\td_embedding\theads\thead_dim\tattn_dropout\t"
        "lr\tcurrent_lr\tscheduler\tscheduler_step\twarmup_epochs\tmin_lr\tweight_decay\t"
        "grad_accum_steps\tema_decay\tbest_split\tbest\ttrain_eval\tvalid10\tvalid90\t"
        "test\tcheckpoint\tmodel_load_path\tcomplete_structures_only\tduration_sec\n"
    )
    log_line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t"
        f"{config.epoch}\t{config.batch}\t{config.mode}\t{config.integer_sequence}\t"
        f"{config.d_hidden}\t{config.dim}\t{config.d_embedding}\t{config.n_heads}\t{config.head_dim}\t{config.attn_dropout}\t"
        f"{config.learning_rate}\t{current_lr:.6f}\t{config.scheduler}\t{config.scheduler_step}\t{config.warmup_epochs}\t{config.min_lr}\t{config.weight_decay}\t"
        f"{config.grad_accum_steps}\t{config.ema_decay}\t{config.best_metric_split}\t{_fmt(run_best_metric)}\t"
        f"{_fmt(train_eval_last)}\t{_fmt(valid10_last)}\t{_fmt(valid90_last)}\t"
        f"{_fmt(test_loss_val)}\t{final_save_path or 'none'}\t"
        f"{config.model_load_path or 'none'}\t{config.complete_structures_only}\t{duration_sec:.1f}\n"
    )
    with open(log_path, 'a') as f:
        if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
            f.write(header)
        f.write(log_line)
    # Plot the loss of each batch over time
    plt.plot(np.sqrt(np.asarray(batch_losses)), label='batch loss')
    plt.ylabel("RMSE")
    plt.xlabel("Step")
    plt.title("Training Loss over Time")
    plt.savefig('TrainingLoss.png')

    # While the above plot demonstrates each batch's loss during training,
    # the plot below shows the performance of the model on several data splits
    # at the *end* of each epoch.
    plt.plot([x.cpu().detach().numpy() for x in epoch_training_losses], label='train-eval')
    plt.plot([x.cpu().detach().numpy() for x in epoch_validation10_losses], label='valid-10')
    plt.plot([x.cpu().detach().numpy() for x in epoch_validation90_losses], label='valid-90')
    plt.ylabel("RMSE")
    plt.xlabel("Epoch")
    plt.title("Training and Validation Losses over Time")
    plt.legend()
    plt.savefig('ValidationLoss.png')
    
    return model, final_save_path
