import torch
import sidechainnet as scn
import sidechainnet.dataloaders.collate as scn_collate
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES
import random
import os
import numpy as np
from collections import OrderedDict

from visualize import build_visualizable_structures, plot_protein
from model import ProteinNet
from trainer import train
from config import get_parameters

# Some SidechainNet CASP12/30 entries contain coordinate arrays with shape
# (L, 14, 3) instead of the expected (L*14, 3), which causes padding to fail.
# Flatten/repair coords before batching. Coordinates are not used by this
# training loop, so simple reshaping/zero-filling is acceptable.
_orig_pad_for_batch = scn_collate.pad_for_batch


def _pad_for_batch_with_coord_fix(items, batch_length, dtype="", *args, **kwargs):
    """Ensure coords are 2D (N, 3) by flattening/repairing before padding."""
    if dtype == "crd":
        fixed_items = []
        # Allow the batch_length to grow if any coord array is longer than expected.
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

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# To train with a GPU, go to Runtime > Change runtime type
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using {device} for training.")
def main(config, dataloader):
    print("Available Dataloaders =", list(dataloader.keys()))

    # Create the model and move it to the GPU
    model = ProteinNet(d_hidden=config.d_hidden,
                            dim=config.dim,
                            d_in=config.d_in,
                            d_embedding=config.d_embedding,
                            heads = config.n_heads,
                            dim_head = config.head_dim,
                            integer_sequence=config.integer_sequence)
    model = model.to(device)
    if config.model_load_path:
        state = torch.load(config.model_load_path, map_location=device)
        state = _maybe_remap_state_dict_for_dropout(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"Warning: loaded with missing keys {missing} and unexpected keys {unexpected}")

    trained_model, save_path = train(model, config, dataloader, device)
    if save_path:
        print(f"Final checkpoint stored at {save_path}")

def plot(idx, dataloader, config, model=None):
    model = model or ProteinNet(d_hidden=config.d_hidden,
                                dim=config.dim,
                                d_in=config.d_in,
                                d_embedding=config.d_embedding,
                                heads = config.n_heads,
                                dim_head = config.head_dim,
                                attn_dropout = config.attn_dropout,
                                integer_sequence=config.integer_sequence)
    model = model.to(device)
    model_path = config.model_load_path or '{}/model_weights.pth'.format(config.model_save_path)
    state = torch.load(model_path, map_location=device)
    state = _maybe_remap_state_dict_for_dropout(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Warning: loaded with missing keys {missing} and unexpected keys {unexpected}")

    os.makedirs('./plots', exist_ok=True)
    s_pred, s_true, local_idx = build_visualizable_structures(
        model, dataloader['train'], config, device, sample_idx=idx)
    s_pred.to_pdb(local_idx, path='./plots/{}_pred.pdb'.format(idx))
    s_true.to_pdb(local_idx, path='./plots/{}_true.pdb'.format(idx))
    plot_protein('./plots/{}_pred.pdb'.format(idx),
                 './plots/{}_true.pdb'.format(idx),
                 html_path='./plots/{}_compare.html'.format(idx),
                 show=False,
                 label_pred="Predicted",
                 label_true="Ground Truth")

if __name__ == '__main__':
    config = get_parameters()
    print("Model Configuration: ")
    print(config)
    # Load the data in the appropriate format for training.
    dataloader = scn.load(
                with_pytorch="dataloaders",
                batch_size=config.batch, 
                dynamic_batching=False,
                num_workers=config.num_workers,
                complete_structures_only=config.complete_structures_only)
    if config.train:
        main(config, dataloader)
    else:
        # If idx is negative, run a small batch of indices (default 0-19)
        if config.idx < 0:
            for i in range(20):
                print(f"Inference on idx={i}")
                plot(i, dataloader, config)
        else:
            # Run inference for a range of indices if idx is negative
            if config.idx < 0:
                html_files = []
                os.makedirs('./plots', exist_ok=True)
                for i in range(0, 100):
                    print(f"Inference on idx={i}")
                    plot(i, dataloader, config)
                    compare_path = f"{i}_compare.html"
                    if os.path.exists(os.path.join('plots', compare_path)):
                        html_files.append(compare_path)
                # Create a combined index html that embeds all compare files
                index_path = os.path.join('plots', 'all_compare.html')
                entries = "\n".join([f'<iframe src="{name}" style="width:100%;height:700px;border:none;margin-bottom:16px;"></iframe>' for name in html_files])
                index_html = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>All Comparisons</title></head>
<body style="margin:0;padding:16px;background:#f5f7fa;font-family:Arial,sans-serif;">
<h2>All comparisons (idx 0-99)</h2>
{entries}
</body>
</html>"""
                with open(index_path, 'w') as f:
                    f.write(index_html)
                print(f"Combined HTML written to {index_path}")
            else:
                plot(config.idx, dataloader, config)
