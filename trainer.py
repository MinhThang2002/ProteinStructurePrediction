import os
import time
import torch
from tqdm import tqdm
import numpy as np
import sidechainnet as scn
from helper import init_loss_optimizer
import matplotlib.pyplot as plt

def validation(model, datasplit, device, loss_fn, mode):
    """Evaluate a model (sequence->sin/cos represented angles [-1,1]) on MSE."""
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
    optimizer, batch_losses, epoch_training_losses, epoch_validation10_losses, epoch_validation90_losses, mse_loss = init_loss_optimizer(model, config)
    os.makedirs(config.model_save_path, exist_ok=True)
    best_metric_file = os.path.join(config.model_save_path, 'best_metric.txt')
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
    for epoch in range(config.epoch):
        print(f'Epoch {epoch}')
        progress_bar = tqdm(total=len(dataloader['train']), smoothing=0)
        for batch in dataloader['train']:
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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Housekeeping
            batch_losses.append(float(loss))
            progress_bar.update(1)
            progress_bar.set_description(f"\rRMSE Loss = {np.sqrt(float(loss)):.4f}")
        # Evaluate the model's performance on train-eval, downsampled for efficiency
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
            torch.save(model.state_dict(), run_best_path)
            print(f"     Updated run-best ({config.best_metric_split}) = {run_best_metric:.4f}")
    # Finally, evaluate the model on the test set
    print(f"Test loss = {validation(model, dataloader['test'], device, mse_loss, config.mode):.4f}")
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
