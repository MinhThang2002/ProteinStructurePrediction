import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model Hyper-parameters
    parser.add_argument('--d_in', dest='d_in', default=49, type=int,
                        help="Model input dimension.")
    parser.add_argument('--d_hidden', dest='d_hidden', default=512, type=int,
                        help="Dimensionality of RNN hidden state.")
    parser.add_argument('--dim', dest='dim', default=256, type=int,
                        help="Attention Layer Dim.")
    parser.add_argument('--d_embedding', dest='d_embedding', default=32, type=int,
                        help="Embedding dimension.")
    parser.add_argument('--n_heads', dest='n_heads', default=8, type=int,
                        help="Number of heads in Attention Layer.")
    parser.add_argument('-h_dim','--head_dim', dest='head_dim', default=64, type=int,
                        help="Dimension of heads in Attention Layer.")
    parser.add_argument('-int_seq','--integer_sequence', dest='integer_sequence', type=str2bool, default=False,
                        help="Dimension of heads in Attention Layer.")

    # Training parameters
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', default=0.001, type=float,
                        help="Learning Rate.")
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.0, type=float,
                        help="Weight decay for optimizer.")
    parser.add_argument('-e', '--epoch', type=int, default=10, help="Training Epochs.")
    parser.add_argument('-b', '--batch', dest='batch',type=int, default=4,
                        help="Batch size during each training step.")
    parser.add_argument('-t','--train', type=str2bool, default=False,help="True when train the model, \
                        else used for testing.")
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0,
                        help="Dataloader workers. Use 0 on macOS/Windows to avoid collate pickling issues.")
    parser.add_argument('--mode', type=str, default='pssms', choices=['pssms', 'seqs'],
                        help="Mode of trainig the model. Select the input of model either to be PSSM-Position Specific Scoring Matrix \
                              or Seqs(Protein Sequence)")
    # Validation
    parser.add_argument('--idx', type=int, default=0,
                        help="Validation index")
    parser.add_argument('--complete_structures_only', dest='complete_structures_only',
                        type=str2bool, default=False,
                        help="If True, only load proteins without missing residues (useful for clean visualization).")
    parser.add_argument('--attn_dropout', dest='attn_dropout', default=0.0, type=float,
                        help="Dropout probability inside attention.")
    parser.add_argument('--model_load_path', dest='model_load_path', type=str, default=None,
                        help="Optional path to load existing model_weights.pth for fine-tuning.")
    parser.add_argument('--save_best', dest='save_best', type=str2bool, default=True,
                        help="Save checkpoint with best validation metric during training.")
    parser.add_argument('--best_metric_split', dest='best_metric_split',
                        choices=['train-eval', 'valid-10', 'valid-90'], default='valid-90',
                        help="Which split to monitor for best checkpoint.")

    # Base Directory
    parser.add_argument('-m', '--model_save_path', type=str, default='./models',
                        help="Path to Saved model directory.")
    return parser.parse_args()
    
