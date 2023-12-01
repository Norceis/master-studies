import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_dim, output_size, num_layers, dropout
    ):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        self.dropout_layer = nn.Dropout(dropout)
        self.embedding_layer = nn.Embedding(input_size, embedding_size)
        self.lstm_layer = nn.LSTM(
            embedding_size, hidden_dim, num_layers, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.embedding_layer(x)
        x = self.dropout_layer(x)
        outputs, (hidden, cell) = self.lstm_layer(x)
        outputs = self.fc(outputs)
        outputs = outputs.squeeze(0)

        return outputs


class MyLSTM(nn.Module):
    def __init__(self, decoder, vocab_size):
        super(MyLSTM, self).__init__()
        self.decoder = decoder
        self.vocab_size = vocab_size

    def forward(self, tgt, tfr=0.1):
        batch_size = tgt.shape[1]  # check this
        target_len = tgt.shape[0]
        outputs = torch.zeros(target_len, batch_size, self.vocab_size).to(device)

        x = tgt[0]
        for t in range(1, target_len):
            output = self.decoder(x)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = tgt[t] if random.random() < tfr else best_guess

        return outputs
