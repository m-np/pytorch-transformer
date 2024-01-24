
# torch packages
import torch.nn as nn
from sublayers import (MultiHeadAttention, PositionwiseFeedForward)


class EncoderLayer(nn.Module):
    """
    This building block in the encoder layer consists of the following
    1. MultiHead Attention
    2. Sublayer Logic
    3. Positional FeedForward Network
    """

    def __init__(self, dk, dv, h, dim_multiplier=4, pdropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dk, dv, h, pdropout)
        # Reference page 5 chapter 3.2.2 Multi-head attention
        dmodel = dk * h
        # Reference page 5 chapter 3.3 positionwise FeedForward
        dff = dmodel * dim_multiplier
        self.attn_norm = nn.LayerNorm(dmodel)
        self.ff = PositionwiseFeedForward(dmodel, dff, pdropout=pdropout)
        self.ff_norm = nn.LayerNorm(dmodel)

        self.dropout = nn.Dropout(p=pdropout)

    def forward(self, src_inputs, src_mask=None):
        """
        Forward pass as per page 3 chapter 3.1
        """
        mha_out, attention_wts = self.attention(
            query=src_inputs, key=src_inputs, val=src_inputs, mask=src_mask
        )

        # Residual connection between input and sublayer output, details: Page 7, Chapter 5.4 "Regularization",
        # Actual paper design is the following
        intermediate_out = self.attn_norm(src_inputs + self.dropout(mha_out))

        pff_out = self.ff(intermediate_out)

        # Perform Add Norm again
        out = self.ff_norm(intermediate_out + self.dropout(pff_out))
        return out, attention_wts


class Encoder(nn.Module):
    def __init__(self, dk, dv, h, num_encoders, dim_multiplier=4, pdropout=0.1):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(dk, dv, h, dim_multiplier, pdropout)
                for _ in range(num_encoders)
            ]
        )

    def forward(self, src_inputs, src_mask=None):
        """
        Input from the Embedding layer
        src_inputs = (B - batch size, S/T - max token sequence length, D- model dimension)
        """
        src_representation = src_inputs

        # Forward pass through encoder stack
        for enc in self.encoder_layers:
            src_representation, attn_probs = enc(src_representation, src_mask)

        self.attn_probs = attn_probs
        return src_representation
