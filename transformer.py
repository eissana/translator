import torch
import torch.nn as nn


class Head(nn.Module):
    """
    Self-attention head layer.
    """

    def __init__(self, head_size, params, use_mask):
        super().__init__()

        embedding_dim = params["embedding_dim"]
        block_size = params["block_size"]

        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(params["dropout"])

        self.use_mask = use_mask
        if use_mask:
            # tril is not a model parameter so we register it as a buffer.
            # block_size is the maximum size. The actual size can be smaller.
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, v, k, q):
        _, T, C = q.shape
        value, key, query = self.value(v), self.key(k), self.query(q)
        weights = query @ key.transpose(-2, -1) * C**-0.5

        if self.use_mask:
            # The time dimension can be smaller than the block-size.
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        out = weights @ value
        return out


class MultiHead(nn.Module):
    def __init__(self, params, use_mask, device):
        super().__init__()
        self.device = device

        embedding_dim = params["embedding_dim"]
        nhead = params["nhead"]
        assert (
            embedding_dim % nhead == 0
        ), f"{embedding_dim=} must be divisible by {nhead=}"
        head_size = embedding_dim // nhead

        self.ln = nn.LayerNorm(embedding_dim)
        self.heads = nn.ModuleList(
            [Head(head_size, params, use_mask) for _ in range(nhead)]
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(params["dropout"])

    def forward(self, v, k, q):
        v, k, q = self.ln(v), self.ln(k), self.ln(q)
        out = torch.cat([head(v, k, q) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, params):
        super().__init__()

        embedding_dim = params["embedding_dim"]
        dim_feedforward = params["dim_feedforward"]

        # feed-forward network
        self.ffn = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, dim_feedforward * embedding_dim),
            nn.ReLU(),
            nn.Linear(dim_feedforward * embedding_dim, embedding_dim),  # projection
            nn.Dropout(params["dropout"]),
        )

    def forward(self, x):
        out = self.ffn(x)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, params, device):
        super().__init__()

        embedding_dim = params["embedding_dim"]
        nhead = params["nhead"]

        # multi-head self attention with no mask. All nodes are allowed to
        # communicate freely.
        self.attn = MultiHead(params, use_mask=False, device=device)
        self.ffn = FeedForward(params)

    def forward(self, v, k, q):
        out = q + self.attn(v, k, q)
        out = out + self.ffn(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, params, device):
        super().__init__()

        embedding_dim = params["embedding_dim"]

        # multi-head self attention with triangular mask. Nodes communicate only
        # with previous nodes.
        self.attn = MultiHead(params, use_mask=True, device=device)
        # Reusing Encoder as the top part of the decoder with a multi-head
        # cross-attention and a feed-forward network on top of it.
        self.attn_ffn = EncoderBlock(params, device)

    def forward(self, enc_out, dec_in):
        out = dec_in
        out = out + self.attn(out, out, out)
        out = out + self.attn_ffn(enc_out, enc_out, out)
        return out


class CustomTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, device, params):
        super().__init__()
        self.device = device

        embedding_dim = params["embedding_dim"]
        block_size = params["block_size"]
        num_encoder_layers = params["num_encoder_layers"]
        num_decoder_layers = params["num_decoder_layers"]
        dropout = params["dropout"]

        self.src_emb = nn.Embedding(src_vocab_size, embedding_dim)
        self.src_pos = nn.Embedding(block_size, embedding_dim)

        self.tgt_emb = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.tgt_pos = nn.Embedding(block_size, embedding_dim)

        self.encoders = nn.ModuleList(
            [EncoderBlock(params, device) for _ in range(num_encoder_layers)]
        )
        self.decoders = nn.ModuleList(
            [DecoderBlock(params, device) for _ in range(num_decoder_layers)]
        )

        self.proj = nn.Linear(embedding_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        _, srcT = src.shape
        src_positions = torch.arange(srcT).unsqueeze(0).to(self.device)
        src_out = self.src_emb(src) + self.src_pos(src_positions)
        src_out = self.dropout(src_out)

        _, tgtT = tgt.shape
        tgt_positions = torch.arange(tgtT).unsqueeze(0).to(self.device)
        tgt_out = self.tgt_emb(tgt) + self.tgt_pos(tgt_positions)
        tgt_out = self.dropout(tgt_out)

        for encoder in self.encoders:
            src_out = encoder(src_out, src_out, src_out)

        for decoder in self.decoders:
            tgt_out = decoder(src_out, tgt_out)

        tgt_out = self.proj(tgt_out)
        tgt_out = self.dropout(tgt_out)

        return tgt_out


class Transformer(nn.Module):
    """
    Using pytorch built-in transformer.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        num_encoder_layers,
        num_decoder_layers,
        src_pad_idx,
        tgt_pad_idx,
        block_size,
        device,
        embedding_dim,
        nhead,
        dim_feedforward,
        dropout,
    ):
        super().__init__()
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_emb = nn.Embedding(src_vocab_size, embedding_dim)
        self.src_pos = nn.Embedding(block_size, embedding_dim)

        self.tgt_emb = nn.Embedding(tgt_vocab_size, embedding_dim)
        self.tgt_pos = nn.Embedding(block_size, embedding_dim)

        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.proj = nn.Linear(embedding_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        _, srcT = src.shape
        src_positions = torch.arange(srcT).unsqueeze(0).to(self.device)
        src_out = self.src_emb(src) + self.src_pos(src_positions)
        src_out = self.dropout(src_out)

        _, tgtT = tgt.shape
        tgt_positions = torch.arange(tgtT).unsqueeze(0).to(self.device)
        tgt_out = self.tgt_emb(tgt) + self.tgt_pos(tgt_positions)
        tgt_out = self.dropout(tgt_out)

        # Avoiding unnecessary computation on padded text
        src_key_padding_mask = src == self.src_pad_idx
        tgt_key_padding_mask = tgt == self.tgt_pad_idx

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgtT).to(self.device)
        out = self.transformer(
            src=src_out,
            tgt=tgt_out,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )

        out = self.proj(out)

        return out
