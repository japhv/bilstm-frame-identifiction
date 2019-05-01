import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab, label_size, device, dropout=0.5, attention_layer=False):
        super(BiLSTM, self).__init__()
        self.attention_layer = attention_layer
        self.hidden_dim = hidden_dim
        self.device = device
        self.dropout = dropout
        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.embeddings.from_pretrained(vocab.vectors, freeze=True, sparse=True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, dropout=dropout, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)

    def init_hidden(self, x):
        # first is the hidden h
        # second is the cell c
        return (autograd.Variable(torch.zeros(2, x.size(1), self.hidden_dim).to(self.device)),
                autograd.Variable(torch.zeros(2, x.size(1), self.hidden_dim).to(self.device)))

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        rnn_out = rnn_out.transpose(0,1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = F.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

    def forward(self, sentence):
        x = self.embeddings(sentence)
        hidden = self.init_hidden(sentence)
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm(x, hidden)
        if self.attention_layer:
            y = self.attention(lstm_out, final_hidden_state)
        else:
            y = lstm_out[-1]

        y = self.hidden2label(y)
        return y, F.sigmoid(y)
