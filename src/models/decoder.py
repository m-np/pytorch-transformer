# torch packages
import torch.nn as nn
from sublayers import MultiHeadAttention, PositionwiseFeedForward
from torch import Tensor


class DecoderLayer(nn.Module):
    def __init__(self, dk, dv, h, dim_multiplier=4, pdropout=0.1):
        super().__init__()

        # Reference page 5 chapter 3.2.2 Multi-head attention
        dmodel = dk * h
        # Reference page 5 chapter 3.3 positionwise FeedForward
        dff = dmodel * dim_multiplier

        # Masked Multi Head Attention
        self.masked_attention = MultiHeadAttention(dk, dv, h, pdropout)
        self.masked_attn_norm = nn.LayerNorm(dmodel)

        # Multi head attention
        self.attention = MultiHeadAttention(dk, dv, h, pdropout)
        self.attn_norm = nn.LayerNorm(dmodel)

        # Add position FeedForward Network
        self.ff = PositionwiseFeedForward(dmodel, dff, pdropout=pdropout)
        self.ff_norm = nn.LayerNorm(dmodel)

        self.dropout = nn.Dropout(p=pdropout)

    def forward(self, trg: Tensor, src: Tensor, trg_mask: Tensor, src_mask: Tensor):
        """
        Args:
            trg:          embedded sequences                (batch_size, trg_seq_length, d_model)
            src:          embedded sequences                (batch_size, src_seq_length, d_model)
            trg_mask:     mask for the sequences            (batch_size, 1, trg_seq_length, trg_seq_length)
            src_mask:     mask for the sequences            (batch_size, 1, 1, src_seq_length)

        Returns:
            trg:          sequences after self-attention    (batch_size, trg_seq_length, d_model)
            attn_probs:   self-attention softmax scores     (batch_size, n_heads, trg_seq_length, src_seq_length)
        """
        _trg, attn_probs = self.masked_attention(
            query=trg, key=trg, val=trg, mask=trg_mask
        )

        # Residual connection between input and sublayer output, details: Page 7, Chapter 5.4 "Regularization",
        # Actual paper design is the following
        trg = self.masked_attn_norm(trg + self.dropout(_trg))

        # Inputs to the decoder attention is given as follows
        # query = previous decoder layer
        # key and val = output of encoder
        # mask = src_mask
        # Reference : page 5 chapter 3.2.3 point 1
        _trg, attn_probs = self.attention(query=trg, key=src, val=src, mask=src_mask)
        trg = self.attn_norm(trg + self.dropout(_trg))

        # position-wise feed-forward network
        _trg = self.ff(trg)
        # Perform Add Norm again
        trg = self.ff_norm(trg + self.dropout(_trg))
        return trg, attn_probs


class Decoder(nn.Module):
    def __init__(self, dk, dv, h, num_decoders, dim_multiplier=4, pdropout=0.1):
        super().__init__()
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(dk, dv, h, dim_multiplier, pdropout)
                for _ in range(num_decoders)
            ]
        )

    def forward(self, target_inputs, src_inputs, target_mask, src_mask):
        """
        Input from the Embedding layer
        target_inputs = embedded sequences    (batch_size, trg_seq_length, d_model)
        src_inputs = embedded sequences       (batch_size, src_seq_length, d_model)
        target_mask = mask for the sequences  (batch_size, 1, trg_seq_length, trg_seq_length)
        src_mask = mask for the sequences     (batch_size, 1, 1, src_seq_length)
        """
        target_representation = target_inputs

        # Forward pass through decoder stack
        for layer in self.decoder_layers:
            target_representation, attn_probs = layer(
                target_representation, src_inputs, target_mask, src_mask
            )
        self.attn_probs = attn_probs
        return target_representation
