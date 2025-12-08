import os
import torch
import numpy as np
import time
from collections import OrderedDict
import sidechainnet as scn
import sidechainnet.dataloaders.collate as scn_collate
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES

from model import ProteinNet
from trainer import validation
from config import get_parameters

# Patch collate to handle (L,14,3) coords like main.py
_orig_pad_for_batch = scn_collate.pad_for_batch


def _pad_for_batch_with_coord_fix(items, batch_length, dtype="", *args, **kwargs):
    """Ensure coords are 2D (N, 3) by flattening/repairing before padding."""
    if dtype == "crd":
        fixed_items = []
        max_len = batch_length or 0
        for item in items:
            arr = np.asarray(item)
            if arr.ndim == 3:
                arr = arr.reshape(-1, arr.shape[-1])  # (L,14,3) -> (L*14,3)
            elif arr.ndim == 2 and arr.shape[-1] != 3:
                arr = np.zeros((arr.shape[0] * arr.shape[1], 3), dtype=arr.dtype)
            fixed_items.append(arr)
            res_len = int(np.ceil(arr.shape[0] / NUM_COORDS_PER_RES))
            if res_len > max_len:
                max_len = res_len
        batch_length = max_len
        items = fixed_items
    return _orig_pad_for_batch(items, batch_length, dtype, *args, **kwargs)


scn_collate.pad_for_batch = _pad_for_batch_with_coord_fix


def _maybe_remap_state_dict_for_dropout(state_dict):
    """Remap older checkpoints (without projection dropout) to new key names."""
    if not isinstance(state_dict, OrderedDict):
        state_dict = OrderedDict(state_dict)
    remaps = [
        ("hidden2out.2.weight", "hidden2out.3.weight"),
        ("hidden2out.2.bias", "hidden2out.3.bias"),
        ("final.1.weight", "final.2.weight"),
        ("final.1.bias", "final.2.bias"),
    ]
    for old, new in remaps:
        if old in state_dict and new not in state_dict:
            state_dict[new] = state_dict.pop(old)
    return state_dict


def run_eval(config):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for evaluation.")

    # Data
    dataloader = scn.load(
        with_pytorch="dataloaders",
        batch_size=config.batch,
        dynamic_batching=False,
        num_workers=config.num_workers,
        complete_structures_only=config.complete_structures_only,
    )

    # Model
    model = ProteinNet(
        d_hidden=config.d_hidden,
        dim=config.dim,
        d_in=config.d_in,
        d_embedding=config.d_embedding,
        heads=config.n_heads,
        dim_head=config.head_dim,
        attn_dropout=config.attn_dropout,
        integer_sequence=config.integer_sequence,
    ).to(device)

    model_path = config.model_load_path or os.path.join(config.model_save_path, "model_weights.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    state = torch.load(model_path, map_location=device)
    state = _maybe_remap_state_dict_for_dropout(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Warning: loaded with missing keys {missing} and unexpected keys {unexpected}")
    model.eval()

    loss_fn = torch.nn.SmoothL1Loss()
    splits = {
        "train-eval": "train-eval",
        "valid-10": "valid-10",
        "valid-90": "valid-90",
        "test": "test",
    }

    results = {}
    for name, key in splits.items():
        rmse = validation(model, dataloader[key], device, loss_fn, config.mode)
        results[name] = float(rmse)

    for name, val in results.items():
        print(f"{name} RMSE = {val:.4f}")

    # Write a simple report
    os.makedirs(config.model_save_path, exist_ok=True)
    report_path = os.path.join(config.model_save_path, "metrics_report.txt")
    header = ("timestamp\tgpu_available\tmode\tinteger_seq\td_hidden\tdim\td_embedding\t"
              "heads\thead_dim\tbatch\tcheckpoint\ttrain_eval\tvalid10\tvalid90\ttest\n")
    with open(report_path, "a") as f:
        if not os.path.exists(report_path) or os.path.getsize(report_path) == 0:
            f.write(header)
        line = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t"
            f"{torch.cuda.is_available()}\t{config.mode}\t{config.integer_sequence}\t"
            f"{config.d_hidden}\t{config.dim}\t{config.d_embedding}\t"
            f"{config.n_heads}\t{config.head_dim}\t"
            f"{config.batch}\t{model_path}\t"
            f"{results['train-eval']:.4f}\t{results['valid-10']:.4f}\t"
            f"{results['valid-90']:.4f}\t{results['test']:.4f}\n"
        )
        f.write(line)
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    cfg = get_parameters()
    run_eval(cfg)
