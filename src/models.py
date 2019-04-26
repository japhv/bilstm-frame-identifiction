import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab, label_size, device, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.dropout = dropout
        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.embeddings.from_pretrained(vocab.vectors, freeze=True, sparse=True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)

    def init_hidden(self, x):
        # first is the hidden h
        # second is the cell c
        return (autograd.Variable(torch.zeros(2, x.size(1), self.hidden_dim).to(self.device)),
                autograd.Variable(torch.zeros(2, x.size(1), self.hidden_dim).to(self.device)))

    def forward(self, sentence):
        x = self.embeddings(sentence)
        hidden = self.init_hidden(sentence)
        lstm_out, hidden = self.lstm(x, hidden)
        y = self.hidden2label(lstm_out[-1])
        return y, F.sigmoid(y)
