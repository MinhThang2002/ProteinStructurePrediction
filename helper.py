import math
import torch
from torch import nn
from inspect import isfunction

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

def init_loss_optimizer(model, config, steps_per_epoch=None):
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = None
    if config.scheduler == 'cosine':
        warmup_epochs = max(0, int(config.warmup_epochs))
        min_lr_scale = config.min_lr / config.learning_rate if config.learning_rate > 0 else 0.0

        if config.scheduler_step == 'batch':
            if not steps_per_epoch:
                raise ValueError("steps_per_epoch is required for batch-level scheduler.")
            warmup_steps = max(0, warmup_epochs * steps_per_epoch)
            total_steps = max(1, config.epoch * steps_per_epoch - warmup_steps)

            def lr_lambda(current_step):
                if warmup_steps > 0 and current_step < warmup_steps:
                    return (current_step + 1) / warmup_steps
                progress = (current_step - warmup_steps) / total_steps
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                return max(min_lr_scale, cosine)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            # Cosine decay with optional warmup, stepped per epoch.
            total_epochs = max(1, config.epoch - warmup_epochs)

            def lr_lambda(current_epoch):
                if warmup_epochs > 0 and current_epoch < warmup_epochs:
                    return (current_epoch + 1) / warmup_epochs
                progress = (current_epoch - warmup_epochs) / total_epochs
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                return max(min_lr_scale, cosine)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    batch_losses = []
    epoch_training_losses = []
    epoch_validation10_losses = []
    epoch_validation90_losses = []
    loss_fn = torch.nn.SmoothL1Loss()
    
    return optimizer, scheduler, batch_losses, epoch_training_losses, epoch_validation10_losses, epoch_validation90_losses, loss_fn
