
# torch packages
import torch
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder
from sublayers import (Embedding, PositionalEncoding)


class Transformer(nn.Module):
    def __init__(
        self,
        dk,
        dv,
        h,
        src_vocab_size,
        target_vocab_size,
        num_encoders,
        num_decoders,
        src_pad_idx,
        target_pad_idx,
        dim_multiplier=4,
        pdropout=0.1,
        device="cpu",
    ):
        super().__init__()

        # Reference page 5 chapter 3.2.2 Multi-head attention
        dmodel = dk * h
        # Modules required to build Encoder
        self.src_embeddings = Embedding(src_vocab_size, dmodel)
        self.src_positional_encoding = PositionalEncoding(
            dmodel, max_seq_length=src_vocab_size, pdropout=pdropout
        )
        self.encoder = Encoder(
            dk, dv, h, num_encoders, dim_multiplier=dim_multiplier, pdropout=pdropout
        )

        # Modules required to build Decoder
        self.target_embeddings = Embedding(target_vocab_size, dmodel)
        self.target_positional_encoding = PositionalEncoding(
            dmodel, max_seq_length=target_vocab_size, pdropout=pdropout
        )
        self.decoder = Decoder(dk, dv, h, num_decoders, dim_multiplier=4, pdropout=0.1)

        # Final output
        self.linear = nn.Linear(dmodel, target_vocab_size)
        #         self.softmax = nn.Softmax(dim=-1)
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.init_params()

    # This part wasn't mentioned in the paper, but it's super important!
    def init_params(self):
        """
        xavier has tremendous impact! I didn't expect
        that the model's perf, with normalization layers,
        is so dependent on the choice of weight initialization.
        """
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        """
        Args:
            src: raw sequences with padding        (batch_size, seq_length)
            src_pad_idx(int): index where the token need not be attended

        Returns:
            src_mask: mask for each sequence            (batch_size, 1, 1, seq_length)
        """
        batch_size = src.shape[0]
        # assign 1 to tokens that need attended to and 0 to padding tokens,
        # then add 2 dimensions
        src_mask = (src != self.src_pad_idx).view(batch_size, 1, 1, -1)
        return src_mask

    def make_target_mask(self, target):
        """
        Args:
            target:  raw sequences with padding        (batch_size, seq_length)
            target_pad_idx(int): index where the token need not be attended

        Returns:
            target_mask: mask for each sequence   (batch_size, 1, seq_length, seq_length)
        """

        seq_length = target.shape[1]
        batch_size = target.shape[0]

        # assign True to tokens that need attended to and
        # False to padding tokens, then add 2 dimensions
        target_mask = (target != self.target_pad_idx).view(
            batch_size, 1, 1, -1
        )  # (batch_size, 1, 1, seq_length)

        # generate subsequent mask
        trg_sub_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=self.device)
        ).bool()  # (batch_size, 1, seq_length, seq_length)

        # bitwise "and" operator | 0 & 0 = 0, 1 & 1 = 1, 1 & 0 = 0
        target_mask = target_mask & trg_sub_mask

        return target_mask

    def forward(self, src_token_ids_batch, target_token_ids_batch):
        # create source and target masks
        src_mask = self.make_src_mask(
            src_token_ids_batch
        )  # (batch_size, 1, 1, src_seq_length)
        target_mask = self.make_target_mask(
            target_token_ids_batch
        )  # (batch_size, 1, trg_seq_length, trg_seq_length)

        # Create embeddings
        src_representations = self.src_embeddings(src_token_ids_batch)
        src_representations = self.src_positional_encoding(src_representations)

        target_representations = self.target_embeddings(target_token_ids_batch)
        target_representations = self.target_positional_encoding(target_representations)

        # Encode
        encoded_src = self.encoder(src_representations, src_mask)

        # Decode
        decoded_output = self.decoder(
            target_representations, encoded_src, target_mask, src_mask
        )

        # Post processing
        out = self.linear(decoded_output)
        # Don't use softmax as we are not comparing against softmaxed output while
        # computing loss. We are comparing against linear outputs
        #         # Output
        #         out = self.softmax(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """
    Following parameters are for Multi30K dataset
    """
    dk = 32
    dv = 32
    h = 8
    src_vocab_size = 7983
    target_vocab_size = 5979
    src_pad_idx = 2
    target_pad_idx = 2
    num_encoders = 3
    num_decoders = 3
    dim_multiplier = 4
    pdropout = 0.1
    # print(111)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        dk,
        dv,
        h,
        src_vocab_size,
        target_vocab_size,
        num_encoders,
        num_decoders,
        dim_multiplier,
        pdropout,
        device=device,
    )
    if torch.cuda.is_available():
        model.cuda()
    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters")
