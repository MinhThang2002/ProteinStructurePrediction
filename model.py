import torch
from torch import nn
from attention_layer import Attention
import sidechainnet as scn

class ProteinNet(nn.Module):
    """A protein sequence-to-angle model that consumes integer-coded sequences."""
    def __init__(self,
                 d_hidden,
                 dim,
                 d_in=21,
                 d_embedding=32,
                 heads = 8,
                 dim_head = 64,
                 integer_sequence=True,
                 n_angles=scn.structure.build_info.NUM_ANGLES,
                 dropout=0.1,
                 use_positional_encoding=True,
                 max_len=1024):
        
        super(ProteinNet, self).__init__()
        # Dimensionality of RNN hidden state
        self.d_hidden = d_hidden
        self.use_positional_encoding = use_positional_encoding
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
      
        self.attn = Attention(dim = dim, 
                                heads = heads,
                                dim_head = dim_head,
                                dropout = dropout)
        # Output vector dimensionality (per amino acid)
        self.d_out = n_angles * 2
        # Output projection layer. (from RNN -> target tensor)
        self.hidden2out = nn.Sequential(
                            nn.Linear(d_embedding, d_hidden),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_hidden, self.d_out)
                                    )
        self.out2attn = nn.Linear(self.d_out, dim)
        self.final = nn.Sequential(
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim, self.d_out))
        self.norm_0 = nn.LayerNorm([dim])
        self.norm_1 = nn.LayerNorm([dim])
        self.activation_0 = nn.GELU()
        self.activation_1 = nn.GELU()

        # Activation function for the output values (bounds values to [-1, 1])                                  
        self.output_activation = torch.nn.Tanh()

        # We embed our model's input differently depending on the type of input
        self.integer_sequence = integer_sequence
        if self.integer_sequence:
            self.input_embedding = torch.nn.Embedding(d_in, d_embedding, padding_idx=20)
        else:
            self.input_embedding = torch.nn.Linear(d_in, d_embedding)

    def _sinusoidal_pos_encoding(self, seq_len, dim, device):
        """Create sinusoidal positional encodings."""
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def get_lengths(self, sequence, mask=None):
        """Compute the lengths of each sequence in the batch."""
        if mask is not None:
            # Prefer the provided mask (1 for real tokens, 0 for padding)
            lengths = mask.sum(dim=1)
        elif self.integer_sequence:
            lengths = sequence.shape[-1] - (sequence == 20).sum(axis=1)
        else:
            lengths = sequence.shape[1] - (sequence == 0).all(axis=-1).sum(axis=1)
        return lengths.cpu()

    def forward(self, sequence, mask=None):
        """Run one forward step of the model."""
        # First, we compute sequence lengths
        lengths = self.get_lengths(sequence, mask=mask)
        lengths = torch.clamp(lengths, min=1)  # avoid zero-length packs
        original_len = sequence.shape[1]

        # Embed inputs
        output = self.input_embedding(sequence)

        # Add positional encoding if enabled
        if self.use_positional_encoding:
            pe_len = min(original_len, self.max_len)
            pe = self._sinusoidal_pos_encoding(pe_len, output.shape[-1], output.device)
            pe = pe.unsqueeze(0)  # (1, L, d_embedding)
            output[:, :pe_len, :] = output[:, :pe_len, :] + pe

        output = self.dropout(output)

        # Linear projection to attention dimension
        output = self.hidden2out(output)
        output = self.out2attn(output)
        output = self.activation_0(output)
        output = self.norm_0(output)
        output = self.attn(output, mask=mask)
        output = self.activation_1(output)
        output = self.norm_1(output)
        output = self.final(output)
      
        # Next, we need to bound the output values between [-1, 1]
        output = self.output_activation(output)

        # Finally, reshape the output to be (batch, length, angle, (sin/cos val))
        output = output.view(output.shape[0], output.shape[1], 12, 2)

        return output
