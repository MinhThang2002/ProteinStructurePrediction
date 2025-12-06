import torch
import sidechainnet as scn
import sidechainnet.dataloaders.collate as scn_collate
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES
import random
import os
import numpy as np

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
        model.load_state_dict(torch.load(config.model_load_path))

    trained_model, save_path = train(model, config, dataloader, device)
    if save_path:
        print(f"Final checkpoint stored at {save_path}")

def plot(idx, dataloader, config):
    model =  ProteinNet(d_hidden=config.d_hidden,
                        dim=config.dim,
                        d_in=config.d_in,
                        d_embedding=config.d_embedding,
                        heads = config.n_heads,
                        dim_head = config.head_dim,
                        attn_dropout = config.attn_dropout,
                        integer_sequence=config.integer_sequence)
    model = model.to(device)
    model_path = config.model_load_path or '{}/model_weights.pth'.format(config.model_save_path)
    model.load_state_dict(torch.load(model_path))

    if os.path.exists('./plots')==False:
        os.mkdir('./plots')
    s_pred, s_true, local_idx = build_visualizable_structures(
        model, dataloader['train'], config, device, sample_idx=idx)
    s_pred.to_pdb(local_idx, path='./plots/{}_pred.pdb'.format(idx))
    s_true.to_pdb(local_idx, path='./plots/{}_true.pdb'.format(idx))
    plot_protein('./plots/{}_pred.pdb'.format(idx),
                 './plots/{}_true.pdb'.format(idx),
                 html_path='./plots/{}_compare.html'.format(idx),
                 show=False)

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
        plot(config.idx, dataloader, config)
