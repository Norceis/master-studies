import json
from copy import deepcopy

import torch
import math
from torch import nn
import torch.nn.functional as F

from src.data_processing import (
    preprocess_and_encode_string,
    pad_zeros,
    decode_string,
    postprocess_string,
)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = x.to(torch.int64)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, drop_prob):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(drop_prob)
        pe_matrix = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe_matrix[:, 0::2] = torch.sin(position * div_term)
        pe_matrix[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe_matrix", pe_matrix.unsqueeze(0))

    def forward(self, x):
        return self.dropout(
            x + self.pe_matrix[:, : x.shape[1], :].requires_grad_(False)
        )


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))  # Learnable scaling factor
        self.beta = nn.Parameter(torch.zeros(1))  # Learnable shifting factor

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(
            d_model, hidden
        )  # First linear layer to expand the dimensionality
        self.linear2 = nn.Linear(
            hidden, d_model
        )  # Second linear layer to reduce the dimensionality back to d_model
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(p=drop_prob)  # Dropout layer

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, drop_prob: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)

        self.w_o = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(drop_prob)
        self.attention_scores = None

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(
            query.shape[0], query.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            key.shape[0], key.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)

        x, self.attention_scores = attention(query, key, value, self.dropout, mask)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.num_heads * self.head_dim)
        )

        return self.w_o(x)


class Residual(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention_block: MultiHeadAttention,
        ff_block: PositionwiseFeedForward,
        drop_prob,
    ):
        super().__init__()

        self.attention = attention_block
        self.ff = ff_block
        self.residuals = nn.ModuleList([Residual(drop_prob) for _ in range(2)])

    def forward(self, x, src_mask=None):
        x = self.residuals[0](x, lambda x: self.attention(x, x, x, src_mask))
        x = self.residuals[1](x, self.ff)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


def attention(q, k, v, dropout=None, mask=None):
    d_k = q.shape[-1]
    attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attention_scores.masked_fill_(mask == 0, -1e9)
    attention_scores = attention_scores.softmax(dim=-1)
    if dropout is not None:
        attention_scores = dropout(attention_scores)
    return (attention_scores @ v), attention_scores


class DecoderLayer(nn.Module):
    def __init__(
        self,
        attention_block: MultiHeadAttention,
        cross_attention_block: MultiHeadAttention,
        ff_block: PositionwiseFeedForward,
        drop_prob,
    ):
        super().__init__()
        self.attention = attention_block
        self.cross_attention = cross_attention_block
        self.ff = ff_block
        self.residuals = nn.ModuleList([Residual(dropout=drop_prob) for _ in range(3)])

    def forward(self, x, enc_output, encoder_mask, decoder_mask):
        x = self.residuals[0](x, lambda x: self.attention(x, x, x, decoder_mask))
        x = self.residuals[1](
            x, lambda x: self.cross_attention(x, enc_output, enc_output, encoder_mask)
        )
        x = self.residuals[2](x, self.ff)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, enc_output, x, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


class LinearSoftmax(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)


class MyTransformer(nn.Module):
    def __init__(
        self, encoder, decoder, src_embed, tgt_embed, src_pe, tgt_pe, linear, metadata
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pe = src_pe
        self.tgt_pe = tgt_pe
        self.linear = linear
        self.metadata = metadata

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pe(src)

        if len(src) != len(src_mask):
            src_mask = src_mask[: len(src)]

        return self.encoder(src, src_mask)

    def decode(self, enc_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pe(tgt)

        if len(tgt) != len(tgt_mask):
            src_mask = src_mask[: len(tgt)]
            tgt_mask = tgt_mask[: len(tgt)]

        if len(tgt) != len(enc_output):
            enc_output = enc_output[: len(tgt)]

        return self.decoder(enc_output, tgt, src_mask, tgt_mask)  # check this

    def project(self, x):
        return self.linear(x)


def build_transformer(
    vocab_size,
    seq_len,
    d_model,
    n_layers,
    n_heads,
    dropout,
    ffn_hidden,
    turn_on_encoder=True,
):
    encoder_layers = []
    for _ in range(n_layers):
        enc_att = MultiHeadAttention(
            d_model=d_model, num_heads=n_heads, drop_prob=dropout
        )
        enc_ff = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=dropout
        )
        encoder_layers.append(
            EncoderLayer(attention_block=enc_att, ff_block=enc_ff, drop_prob=dropout)
        )

    decoder_layers = []
    for _ in range(n_layers):
        dec_att = MultiHeadAttention(
            d_model=d_model, num_heads=n_heads, drop_prob=dropout
        )
        dec_cross_att = MultiHeadAttention(
            d_model=d_model, num_heads=n_heads, drop_prob=dropout
        )
        dec_ff = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=dropout
        )
        decoder_layers.append(
            DecoderLayer(
                attention_block=dec_att,
                cross_attention_block=dec_cross_att,
                ff_block=dec_ff,
                drop_prob=dropout,
            )
        )

    transformer = MyTransformer(
        encoder=Encoder(nn.ModuleList(encoder_layers)) if turn_on_encoder else None,
        decoder=Decoder(nn.ModuleList(decoder_layers)),
        src_embed=Embeddings(d_model=d_model, vocab_size=vocab_size),
        tgt_embed=Embeddings(d_model=d_model, vocab_size=vocab_size),
        src_pe=PositionalEncoding(
            d_model=d_model, max_seq_length=seq_len, drop_prob=dropout
        ),
        tgt_pe=PositionalEncoding(
            d_model=d_model, max_seq_length=seq_len, drop_prob=dropout
        ),
        linear=LinearSoftmax(d_model=d_model, vocab_size=vocab_size),
        metadata={
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "dropout": dropout,
            "ffn_hidden": ffn_hidden,
        },
    )

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def generate_masks(src, tgt):
    src, tgt = src[0], tgt[0]
    src_mask = (src != 0).unsqueeze(0).unsqueeze(0).int()
    tgt_mask = (tgt != 0).unsqueeze(0).unsqueeze(0).int() & tgt_additional_mask(tgt)
    return src_mask, tgt_mask


def tgt_additional_mask(tgt):
    mask = torch.triu(
        torch.ones(1, tgt.size(0), tgt.size(0), device=tgt.device), diagonal=1
    ).type(torch.int)
    return mask == 0


def generate_fairytale_transformer(
    input_text,
    how_many_sentences,
    experiment_number,
    model_name,
    roulette: bool = False,
):
    with open(f"./models/{experiment_number}/vocab.json", "r") as file:
        vocab = json.load(file)

    with open(f"./models/{experiment_number}/reverse_vocab.json", "r") as file:
        reverse_vocab = json.load(file)

    converted_text = preprocess_and_encode_string(input_text, vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimus_prime = torch.load(f"./models/{experiment_number}/{model_name}")
    optimus_prime.eval()
    optimus_prime.to(device)

    context_len = optimus_prime.metadata["seq_len"]
    fairytale = deepcopy(converted_text)

    while True:
        if how_many_sentences == fairytale.count(3):
            break
        if len(fairytale) > 500:
            break

        input_data = fairytale[-context_len:]
        len_input_data = len(input_data)
        if len(input_data) < context_len:
            input_data = pad_zeros(input_data[:-1], context_len, front=True)

        input_tensor = torch.tensor([input_data], dtype=torch.long)

        # src_data = torch.zeros_like(tgt_data)
        input_tensor = input_tensor.to(device)
        enc_mask, dec_mask = generate_masks(input_tensor, input_tensor)

        enc_mask = enc_mask.expand(1, -1, -1).unsqueeze(1).to(device)
        dec_mask = dec_mask.expand(1, -1, -1).unsqueeze(1).to(device)

        encoder_output = torch.zeros(
            (1, optimus_prime.metadata["seq_len"], optimus_prime.metadata["d_model"])
        ).to(device)
        # encoder_output = optimus_prime.encode(input_tensor, enc_mask)
        decoder_output = optimus_prime.decode(
            encoder_output, enc_mask, input_tensor, dec_mask
        )
        proj_output = optimus_prime.project(decoder_output)

        if roulette:
            topk_values, topk_indices = torch.topk(proj_output, k=10, dim=-1)
            softmax_probs = torch.softmax(topk_values[0], dim=-1)
            samples = torch.multinomial(softmax_probs, 1, replacement=True)
            selected_indices = topk_indices.gather(1, samples.unsqueeze(0))
            fairytale.append(
                selected_indices.reshape((1, context_len)).tolist()[0][
                    len_input_data - 1
                ]
            )
        else:
            argmax_values, argmax_indices = torch.max(proj_output, dim=-1)
            fairytale.append(argmax_indices.tolist()[0][len_input_data - 1])

    decoded_sentence = decode_string(fairytale, reverse_vocab)
    postprocessed_sentence = postprocess_string(decoded_sentence)
    return postprocessed_sentence

def generate_fairytale_transformer_cli(
    input_text,
    how_many_sentences,
    experiment_number,
    model_name,
    roulette: bool = False,
):
    with open(f"./transformer/models/{experiment_number}/vocab.json", "r") as file:
        vocab = json.load(file)

    with open(f"./transformer/models/{experiment_number}/reverse_vocab.json", "r") as file:
        reverse_vocab = json.load(file)

    converted_text = preprocess_and_encode_string(input_text, vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimus_prime = torch.load(f"./transformer/models/{experiment_number}/{model_name}")
    optimus_prime.eval()
    optimus_prime.to(device)

    context_len = optimus_prime.metadata["seq_len"]
    fairytale = deepcopy(converted_text)

    while True:
        if how_many_sentences == fairytale.count(3):
            break
        if len(fairytale) > 500:
            break

        input_data = fairytale[-context_len:]
        len_input_data = len(input_data)
        if len(input_data) < context_len:
            input_data = pad_zeros(input_data[:-1], context_len, front=True)

        input_tensor = torch.tensor([input_data], dtype=torch.long)

        # src_data = torch.zeros_like(tgt_data)
        input_tensor = input_tensor.to(device)
        enc_mask, dec_mask = generate_masks(input_tensor, input_tensor)

        enc_mask = enc_mask.expand(1, -1, -1).unsqueeze(1).to(device)
        dec_mask = dec_mask.expand(1, -1, -1).unsqueeze(1).to(device)

        encoder_output = torch.zeros(
            (1, optimus_prime.metadata["seq_len"], optimus_prime.metadata["d_model"])
        ).to(device)
        # encoder_output = optimus_prime.encode(input_tensor, enc_mask)
        decoder_output = optimus_prime.decode(
            encoder_output, enc_mask, input_tensor, dec_mask
        )
        proj_output = optimus_prime.project(decoder_output)

        if roulette:
            topk_values, topk_indices = torch.topk(proj_output, k=10, dim=-1)
            softmax_probs = torch.softmax(topk_values[0], dim=-1)
            samples = torch.multinomial(softmax_probs, 1, replacement=True)
            selected_indices = topk_indices.gather(1, samples.unsqueeze(0))
            fairytale.append(
                selected_indices.reshape((1, context_len)).tolist()[0][
                    len_input_data - 1
                ]
            )
        else:
            argmax_values, argmax_indices = torch.max(proj_output, dim=-1)
            fairytale.append(argmax_indices.tolist()[0][len_input_data - 1])

    decoded_sentence = decode_string(fairytale, reverse_vocab)
    postprocessed_sentence = postprocess_string(decoded_sentence)
    return postprocessed_sentence
