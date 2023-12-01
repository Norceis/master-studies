import json
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data_processing import (
    preprocess_and_encode_string,
    pad_zeros,
    postprocess_string,
    decode_string,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyLSTM(nn.Module):
    def __init__(
        self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5
    ):
        super(MyLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_size)

        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(lstm_out)
        out = out.view(batch_size, -1, self.output_size)
        out = out[:, -1]

        return out, hidden

    def init_hidden(self, batch_size):
        weights = next(self.parameters()).data
        hidden = (
            weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
            weights.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        )

        return hidden


def check_batch_len(data_batch, label_batch, batch_size):
    if data_batch.size(0) < batch_size:
        num_to_pad = batch_size - data_batch.size(0)
        padding = torch.zeros(
            (num_to_pad, data_batch.size(1)), dtype=data_batch.dtype
        ).to(device)
        data_batch = torch.cat((data_batch, padding), dim=0)
        label_padding = torch.zeros(num_to_pad, dtype=label_batch.dtype).to(device)
        label_batch = torch.cat((label_batch, label_padding), dim=0)

    return data_batch, label_batch


def generate_fairytale_lstm(
    input_text,
    context_len,
    how_many_sentences,
    experiment_number,
    model_name,

):
    with open(f'./models/{experiment_number}/vocab.json', 'r') as file:
        vocab = json.load(file)

    with open(f'./models/{experiment_number}/reverse_vocab.json', 'r') as file:
        reverse_vocab = json.load(file)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f"./models/{experiment_number}/{model_name}")
    model.to(device)
    model.eval()

    converted_text = preprocess_and_encode_string(input_text, vocab)
    fairytale = deepcopy(converted_text)

    while True:
        if how_many_sentences == fairytale.count(3):
            break
        if len(fairytale) > 5000:
            break

        input_data = fairytale[-context_len:]
        if len(input_data) < context_len:
            input_data = pad_zeros(input_data[:-1], context_len, front=True)

        input_tensor = torch.tensor([input_data], dtype=torch.long)
        input_tensor = input_tensor.to(device)

        hidden = model.init_hidden(input_tensor.size(0))
        output, _ = model(input_tensor, hidden)
        p = F.softmax(output, dim=1).data.to(device)
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.cpu().numpy().squeeze()
        p = p.cpu().numpy().squeeze()
        word_i = np.random.choice(top_i, p=p / p.sum())
        fairytale.append(word_i)

    decoded_sentence = decode_string(fairytale, reverse_vocab)
    postprocessed_sentence = postprocess_string(decoded_sentence)

    return postprocessed_sentence

def generate_fairytale_lstm_cli(
    input_text,
    context_len,
    how_many_sentences,
    experiment_number,
    model_name,

):
    with open(f'./lstm/models/{experiment_number}/vocab.json', 'r') as file:
        vocab = json.load(file)

    with open(f'./lstm/models/{experiment_number}/reverse_vocab.json', 'r') as file:
        reverse_vocab = json.load(file)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f"./lstm/models/{experiment_number}/{model_name}")
    model.to(device)
    model.eval()

    converted_text = preprocess_and_encode_string(input_text, vocab)
    fairytale = deepcopy(converted_text)

    while True:
        if how_many_sentences == fairytale.count(3):
            break
        if len(fairytale) > 5000:
            break

        input_data = fairytale[-context_len:]
        if len(input_data) < context_len:
            input_data = pad_zeros(input_data[:-1], context_len, front=True)

        input_tensor = torch.tensor([input_data], dtype=torch.long)
        input_tensor = input_tensor.to(device)

        hidden = model.init_hidden(input_tensor.size(0))
        output, _ = model(input_tensor, hidden)
        p = F.softmax(output, dim=1).data.to(device)
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.cpu().numpy().squeeze()
        p = p.cpu().numpy().squeeze()
        word_i = np.random.choice(top_i, p=p / p.sum())
        fairytale.append(word_i)

    decoded_sentence = decode_string(fairytale, reverse_vocab)
    postprocessed_sentence = postprocess_string(decoded_sentence)

    return postprocessed_sentence
