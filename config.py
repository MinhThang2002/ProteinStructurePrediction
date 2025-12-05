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
    parser.add_argument('--weight_decay', dest='weight_decay', default=1e-4, type=float,
                        help="Weight decay (L2 regularization).")
    parser.add_argument('--scheduler', dest='scheduler', default='cosine', choices=['none', 'cosine', 'step'],
                        help="Learning rate scheduler type.")
    parser.add_argument('--step_size', dest='step_size', default=5, type=int,
                        help="Step size for StepLR.")
    parser.add_argument('--gamma', dest='gamma', default=0.5, type=float,
                        help="Decay factor for StepLR.")
    parser.add_argument('--grad_accum_steps', dest='grad_accum_steps', default=1, type=int,
                        help="Accumulate gradients over N steps to simulate larger batch.")
    parser.add_argument('--patience', dest='patience', default=8, type=int,
                        help="Early stopping patience based on valid-10.")
    parser.add_argument('-e', '--epoch', type=int, default=10, help="Training Epochs.")
    parser.add_argument('-b', '--batch', dest='batch',type=int, default=4,
                        help="Batch size during each training step.")
    parser.add_argument('-t','--train', type=str2bool, default=False,help="True when train the model, \
                        else used for testing.")
    parser.add_argument('--mode', type=str, default='pssms', choices=['pssms', 'seqs'],
                        help="Mode of trainig the model. Select the input of model either to be PSSM-Position Specific Scoring Matrix \
                              or Seqs(Protein Sequence)")
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0,
                        help="DataLoader worker processes. Use 0 on macOS to avoid spawn pickling issues.")
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.1,
                        help="Dropout rate for attention/MLP.")
    parser.add_argument('--use_positional_encoding', dest='use_positional_encoding', type=str2bool, default=True,
                        help="Whether to add sinusoidal positional encodings.")
    parser.add_argument('--max_len', dest='max_len', type=int, default=1024,
                        help="Maximum sequence length for positional encoding.")
    # Validation
    parser.add_argument('--idx', type=int, default=0,
                        help="Validation index")

    # Base Directory
    parser.add_argument('-m', '--model_save_path', type=str, default='./models',
                        help="Path to Saved model directory.")
    return parser.parse_args()
    
