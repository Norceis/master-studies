import torch
import math
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, x, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe_matrix = torch.zeros(max_seq_length, x)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, x, 2).float() * -(math.log(10000.0) / x))

        pe_matrix[:, 0::2] = torch.sin(position * div_term)
        pe_matrix[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe_matrix", pe_matrix.unsqueeze(0))

    def forward(self, x):
        return x + self.pe_matrix[:, : x.size(1)]


# Implementation of the actual attention calculations (Q, K, V vectors)
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]  # Dimension of the key vectors (64 for BERT-base)

    # Compute the attention scores by multiplying query and key matrices and scaling by 1/sqrt(d_k)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)

    # If mask is provided, add it to the scaled matrix (used in the decoder)
    if mask is not None:
        scaled += mask

    # Apply the softmax function to compute the attention probabilities
    attention = F.softmax(scaled, dim=-1)

    # Compute the context (output) vector by multiplying attention probabilities with the value matrix
    values = torch.matmul(attention, v)

    return values, attention


# Self-attention mechanism
class MultiHeadAttentionEnc(nn.Module):
    # Compute Q, K, V matrices using the qkv_layer
    # Q, K, and V are query, key, and value matrices for the attention mechanism
    # x: input tensor (batch_size x max_sequence_length x d_model)

    # Reshape the qkv tensor to separate the heads and the Q, K, V matrices
    # qkv: tensor containing Q, K, and V matrices (batch_size x max_sequence_length x num_heads x 3 * head_dim)

    # Split the tensor into Q, K, V matrices
    # q: query matrix (batch_size x num_heads x max_sequence_length x head_dim)
    # k: key matrix (batch_size x num_heads x max_sequence_length x head_dim)
    # v: value matrix (batch_size x num_heads x max_sequence_length x head_dim)

    # Compute the attention values and attention probabilities using the scaled dot-product attention function
    # values: output tensor after applying attention (batch_size x num_heads x max_sequence_length x head_dim)
    # attention: attention probabilities (batch_size x num_heads x max_sequence_length x max_sequence_length)

    # Reshape the output values to match the input shape
    # values: reshaped output tensor (batch_size x max_sequence_length x d_model)

    # Combine the multi-head outputs using the linear_layer
    # out: final output tensor (batch_size x max_sequence_length x d_model)

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Single linear layer to compute Q, K, V matrices in one shot
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)

        # Output linear layer to combine the multi-head outputs
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, max_sequence_length, d_model = x.size()

        # Compute Q, K, V matrices using the qkv_layer
        qkv = self.qkv_layer(x)

        # Reshape the qkv tensor to separate the heads and the Q, K, V matrices
        qkv = qkv.reshape(
            batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim
        )
        qkv = qkv.permute(0, 2, 1, 3)

        # Split the tensor into Q, K, V matrices
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute the attention values and attention probabilities using the scaled dot-product attention function
        values, attention = scaled_dot_product(q, k, v, mask)

        # Reshape the output values to match the input shape
        values = values.reshape(
            batch_size, max_sequence_length, self.num_heads * self.head_dim
        )

        # Combine the multi-head outputs using the linear_layer
        out = self.linear_layer(values)

        return out


# Layer normalization
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(
            torch.ones(parameters_shape)
        )  # Learnable scaling factor
        self.beta = nn.Parameter(
            torch.zeros(parameters_shape)
        )  # Learnable shifting factor

    def forward(self, inputs):
        # Compute the mean and standard deviation along the last n dimensions
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()

        # Normalize the inputs using the computed mean and standard deviation
        y = (inputs - mean) / std

        # Apply the learnable scaling (gamma) and shifting (beta) factors
        out = self.gamma * y + self.beta

        return out


# Position-wise feed-forward network
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
        # Apply the first linear layer
        x = self.linear1(x)

        # Apply the activation function
        x = self.relu(x)

        # Apply the dropout layer
        x = self.dropout(x)

        # Apply the second linear layer
        x = self.linear2(x)

        return x


# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttentionEnc(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask=None):
        residual_x = x

        # Apply the self-attention mechanism
        x = self.attention(x, mask=mask)

        # Add the residual connection and apply layer normalization
        x = self.norm1(self.dropout1(x) + residual_x)
        residual_x = x

        # Apply the position-wise feed-forward network
        x = self.ffn(x)

        # Add the residual connection and apply layer normalization
        x = self.norm2(self.dropout2(x) + residual_x)

        return x


# Encoder
class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()

        # Stack the encoder layers
        self.layers = SequentialEncoder(
            *[
                EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None):
        # Pass the input through the stacked encoder layers
        x = self.layers(x, mask)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)  # 1536
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size()  # 30 x 200 x 512

        qkv = self.qkv_layer(x)  # 30 x 200 x 1536
        qkv = qkv.reshape(
            batch_size, sequence_length, self.num_heads, 3 * self.head_dim
        )  # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 192
        q, k, v = qkv.chunk(
            3, dim=-1
        )  # q: 30 x 8 x 200 x 64, k: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        values, attention = scaled_dot_product(
            q, k, v, mask
        )  # values: 30 x 8 x 200 x 64
        values = values.reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )  # 30 x 200 x 512
        out = self.linear_layer(values)  # 30 x 200 x 512

        return out  # 30 x 200 x 512


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)  # 1024
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.size()  # 30 x 200 x 512

        kv = self.kv_layer(x)  # 30 x 200 x 1024
        q = self.q_layer(y)  # 30 x 200 x 512
        kv = kv.reshape(
            batch_size, sequence_length, self.num_heads, 2 * self.head_dim
        )  # 30 x 200 x 8 x 128
        q = q.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )  # 30 x 200 x 8 x 64
        kv = kv.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 128
        q = q.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 64
        k, v = kv.chunk(2, dim=-1)  # K: 30 x 8 x 200 x 64, v: 30 x 8 x 200 x 64
        values, attention = scaled_dot_product(q, k, v, mask)  # 30 x 8 x 200 x 64

        values = values.reshape(batch_size, sequence_length, d_model)  # 30 x 200 x 512
        out = self.linear_layer(values)  # 30 x 200 x 512

        return out  # 30 x 200 x 512


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(
            d_model=d_model, num_heads=num_heads
        )
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, encoder_mask, decoder_mask):
        _y = y  # 30 x 200 x 512
        y = self.self_attention(y, mask=decoder_mask)  # 30 x 200 x 512
        y = self.dropout1(y)  # 30 x 200 x 512
        y = self.norm1(y + _y)  # 30 x 200 x 512
        _y = y  # 30 x 200 x 512
        y = self.encoder_decoder_attention(x, y, mask=encoder_mask)  # 30 x 200 x 512
        y = self.dropout2(y)
        y = self.norm2(y + _y)  # 30 x 200 x 512
        _y = y  # 30 x 200 x 512
        y = self.ffn(y)  # 30 x 200 x 512
        y = self.dropout3(y)  # 30 x 200 x 512
        y = self.norm3(y + _y)  # 30 x 200 x 512
        return y  # 30 x 200 x 512


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, enc_mask = inputs
        for module in self._modules.values():
            x = module(x, enc_mask)  # 30 x 200 x 512
        return x


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, enc_mask, dec_mask = inputs
        for module in self._modules.values():
            y = module(x, y, enc_mask, dec_mask)  # 30 x 200 x 512
        return y


class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(
            *[
                DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, y, encoder_mask, decoder_mask):
        # x : 30 x 200 x 512
        # y : 30 x 200 x 512
        # mask : 200 x 200
        y = self.layers(x, y, encoder_mask, decoder_mask)
        return y  # 30 x 200 x 512


class MyTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        super(MyTransformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder = Encoder(d_model, d_ff, num_heads, dropout, num_layers)
        self.decoder = Decoder(d_model, d_ff, num_heads, dropout, num_layers)

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(tgt.device)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1
            - torch.triu(
                torch.ones(1, seq_length, seq_length, device=tgt.device), diagonal=1
            )
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedded = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        enc_output = src_embedded
        enc_output = self.encoder(enc_output, src_mask)

        dec_output = tgt_embedded
        dec_output = self.decoder(enc_output, dec_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        output = self.softmax(output)
        return output
