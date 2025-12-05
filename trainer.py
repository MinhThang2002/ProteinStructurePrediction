import torch
from tqdm.auto import tqdm
import numpy as np
import sidechainnet as scn
from helper import init_loss_optimizer
import matplotlib.pyplot as plt
from copy import deepcopy

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

            # Drop invalid (NaN) angles from loss; keep only real values at masked positions.
            angles = torch.nan_to_num(batch.angs, nan=0.0).to(device)
            angle_mask = torch.isfinite(angles)
            valid_mask = angle_mask & mask_.unsqueeze(-1)  # shape: (B, L, 12)

            true_angles_sincosine = scn.structure.trig_transform(angles).to(device)
            mask = valid_mask.unsqueeze(-1).repeat(1, 1, 1, 2)

            # Make predictions
            predicted_angles = model(seqs, mask = mask_)
            # Align lengths defensively in case of padding mismatches on GPU
            max_len = min(predicted_angles.shape[1], true_angles_sincosine.shape[1], mask.shape[1], mask_.shape[1])
            predicted_angles = predicted_angles[:, :max_len]
            true_angles_sincosine = true_angles_sincosine[:, :max_len]
            mask = mask[:, :max_len]

            mask = mask.to(torch.bool)
            loss = loss_fn(predicted_angles[mask], true_angles_sincosine[mask])
            
            total += loss
            n += 1

    return torch.sqrt(total/n)

def train(model, config, dataloader, device):
    optimizer, scheduler, batch_losses, epoch_training_losses, epoch_validation10_losses, epoch_validation90_losses, mse_loss = init_loss_optimizer(model, config)
    grad_accum_steps = max(1, config.grad_accum_steps)
    best_state = None
    best_val = float('inf')
    patience_ctr = 0

    for epoch in range(config.epoch):
        print(f'Epoch {epoch}')
        progress_bar = tqdm(total=len(dataloader['train']),
                             smoothing=0,
                             leave=False,
                             ncols=80)
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader['train'], 1):
            # Prepare variables and create a mask of missing angles (padded with zeros)
            # Note the mask is repeated in the last dimension to match the sin/cos represenation.
            if config.mode == 'seqs':
                seqs = batch.int_seqs.to(device).long()
            elif config.mode == 'pssms':
                seqs = batch.seq_evo_sec.to(device)
            mask_ = batch.msks.to(device)

            # Build a per-angle mask that excludes NaNs and padded positions.
            angles = torch.nan_to_num(batch.angs, nan=0.0).to(device)
            angle_mask = torch.isfinite(angles)
            valid_mask = angle_mask & mask_.unsqueeze(-1)  # shape: (B, L, 12)

            true_angles_sincos = scn.structure.trig_transform(angles).to(device)
            mask = valid_mask.unsqueeze(-1).repeat(1, 1, 1, 2)

            # Make predictions and optimize
            predicted_angles = model(seqs, mask = mask_)
            max_len = min(predicted_angles.shape[1], true_angles_sincos.shape[1], mask.shape[1], mask_.shape[1])
            predicted_angles = predicted_angles[:, :max_len]
            true_angles_sincos = true_angles_sincos[:, :max_len]
            mask = mask[:, :max_len]

            mask = mask.to(torch.bool)
            loss = mse_loss(predicted_angles[mask], true_angles_sincos[mask])
            loss = loss / grad_accum_steps
            loss.backward()

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                optimizer.zero_grad()

            # Housekeeping
            batch_losses.append(float(loss))
            progress_bar.update(1)
        progress_bar.close()
        # Final step flush if dataset size not divisible by grad_accum_steps
        if (step % grad_accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            optimizer.zero_grad()

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

        # Scheduler step per epoch
        if scheduler is not None:
            scheduler.step()

        # Early stopping on valid-10
        current_val = float(epoch_validation10_losses[-1])
        if current_val + 1e-6 < best_val:
            best_val = current_val
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr > config.patience:
                print(f"Early stopping at epoch {epoch} (best valid-10={best_val:.4f})")
                break

    # Finally, evaluate the model on the test set
    print(f"Test loss = {validation(model, dataloader['test'], device, mse_loss, config.mode):.4f}")
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
    
    # Restore best checkpoint if available
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
