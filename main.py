import torch
import sidechainnet as scn
import random
import os
import numpy as np

from visualize import build_visualizable_structures, plot_protein
from model import ProteinNet
from trainer import train
from config import get_parameters
from sidechainnet.dataloaders import collate as scn_collate
from sidechainnet.structure.build_info import NUM_COORDS_PER_RES

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def patch_sidechainnet_collate():
    """Flatten coordinate arrays before padding to avoid mismatched dims."""
    original_pad = scn_collate.pad_for_batch

    def pad_for_batch(items, batch_length, dtype="", seqs_as_onehot=False, vocab=None):
        if dtype == "crd":
            normalized = []
            max_allowed = batch_length * NUM_COORDS_PER_RES
            for item in items:
                arr = np.asarray(item)
                if arr.ndim == 3:
                    arr = arr.reshape(-1, arr.shape[-1])
                if arr.shape[0] > max_allowed:
                    arr = arr[:max_allowed]
                normalized.append(arr)
            items = normalized
        return original_pad(items, batch_length, dtype=dtype, seqs_as_onehot=seqs_as_onehot, vocab=vocab)

    scn_collate.pad_for_batch = pad_for_batch


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
                            integer_sequence=config.integer_sequence,
                            dropout=config.dropout,
                            use_positional_encoding=config.use_positional_encoding,
                            max_len=config.max_len)
    model = model.to(device)

    trained_model = train(model, config, dataloader, device)
    if os.path.exists(config.model_save_path)==False:
        os.mkdir(config.model_save_path)
    torch.save(trained_model.state_dict(), '{}/model_weights.pth'.format(config.model_save_path))

def plot(idx, dataloader, config):
    model =  ProteinNet(d_hidden=config.d_hidden,
                        dim=config.dim,
                        d_in=config.d_in,
                        d_embedding=config.d_embedding,
                        heads = config.n_heads,
                        dim_head = config.head_dim,
                        integer_sequence=config.integer_sequence,
                        dropout=config.dropout,
                        use_positional_encoding=config.use_positional_encoding,
                        max_len=config.max_len)
    model = model.to(device)
    # Always map checkpoints to the active device (CPU on machines without CUDA)
    checkpoint_path = '{}/model_weights.pth'.format(config.model_save_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if os.path.exists('./plots')==False:
        os.mkdir('./plots')
    s_pred, s_true = build_visualizable_structures(model, dataloader['train'], config, device)
    # Add titles so PDB REMARK header is not the default "Untitled".
    s_pred.to_pdb(idx, path='./plots/{}_pred.pdb'.format(idx), title=f'pred_{idx}')
    s_true.to_pdb(idx, path='./plots/{}_true.pdb'.format(idx), title=f'true_{idx}')
    try:
        plot_protein('./plots/{}_pred.pdb'.format(idx), './plots/{}_true.pdb'.format(idx))
    except ImportError as exc:
        # py3Dmol requires an active notebook/JS context; skip gracefully on CLI.
        print(f"Skipping interactive 3D view: {exc}")

if __name__ == '__main__':
    patch_sidechainnet_collate()
    config = get_parameters()
    print("Model Configuration: ")
    print(config)
    # Load the data in the appropriate format for training.
    dataloader = scn.load(
                with_pytorch="dataloaders",
                batch_size=config.batch, 
                dynamic_batching=False,
                num_workers=config.num_workers)
    if config.train:
        main(config, dataloader)
    else:
        plot(config.idx, dataloader, config)
