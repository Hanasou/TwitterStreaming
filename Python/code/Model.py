import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LstmModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, lstm_units, num_hidden, 
        num_layers, num_classes, drop_prob=0.5, use_gpu=False):
        # Initialize stuff
        # I'm not entirely sure if I have to initialize everything 
        # but I'll do it anyways
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.lstm_units = lstm_units
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.drop_prob = drop_prob
        self.use_gpu = use_gpu

        # Initialize layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, lstm_units, num_layers,
                            dropout=drop_prob,
                            batch_first=True)
        self.fc1 = nn.Linear(lstm_units, num_hidden)
        self.dropout = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(num_hidden, num_classes)
    
    def init_hidden(self, batch_size):
        if self.use_gpu:
            hidden = (torch.zeros(self.num_layers, batch_size, self.lstm_units).cuda(),
                    torch.zeros(self.num_layers, batch_size, self.lstm_units).cuda())
        else:
            hidden = (torch.zeros(self.num_layers, batch_size, self.lstm_units),
                    torch.zeros(self.num_layers, batch_size, self.lstm_units))
        return hidden
    
    def forward(self, text, text_lengths):
        batch_size = text.shape[0]
        hidden = self.init_hidden(batch_size)

        # Embedding layer
        embedded = self.embedding(text)
        # Ignore padded elements
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True)
        # Pass through LSTM
        lstm_output, hidden = self.lstm(packed_embedded, hidden)
        # Put output back to original format
        output_unpacked, output_lengths = pad_packed_sequence(lstm_output, batch_first=True)
        # Reformat to pass into linear layer
        output = output_unpacked[:, -1, :]
        # Pass into relu function fc1
        dense1 = F.relu(self.fc1(output))
        # Dropout
        drop = self.dropout(dense1)
        # Pass into fc2 for final output
        preds = self.fc2(drop)
        return preds