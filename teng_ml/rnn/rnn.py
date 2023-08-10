import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    (Bi)LSTM for name classification
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.is_bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        if bidirectional == True:
          self.fc = nn.Linear(hidden_size * 2, num_classes)
        else:
          self.fc = nn.Linear(hidden_size, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.D = 2 if self.is_bidirectional == True else 1

    def forward(self, x, unpadded_lengths=None):
        """
        @param x:
            Tensor (seq_length, features) for unbatched inputs
            Tensor (batch_size, seq_length, features) for batch inputs
            PackedSequence for padded batched inputs
        @param unpadded_lengths: Tensor(batch_size) with lengths of the unpadded sequences, when using padding but without PackedSequence
        @returns (batch_size, num_classes)  with batch_size == 1 for unbatched inputs
        """
        # if type(x) == torch.Tensor:
        #     device = x.device
        #     # h0: initial hidden states
        #     # c0: initial cell states
        #     if len(x.shape) == 2:  # x: (seq_length, features)
        #         h0 = torch.zeros(self.D * self.num_layers, self.hidden_size).to(device)
        #         c0 = torch.zeros(self.D * self.num_layers, self.hidden_size).to(device)
        #     elif len(x.shape) == 3:   # x: (batch, seq_length, features)
        #         batch_size = x.shape[0]
        #         h0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(device)
        #         c0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(device)
        #     else:
        #         raise ValueError(f"RNN.forward: invalid input shape: {x.shape}. Must be (batch, seq_length, features) or (seq_length, features)")
        # elif type(x) == nn.utils.rnn.PackedSequence:
        #     device = x.data.device
        #     h0 = torch.zeros(self.D * self.num_layers, self.hidden_size).to(device)
        #     c0 = torch.zeros(self.D * self.num_layers, self.hidden_size).to(device)
        # else:
        #     raise ValueError(f"RNN.forward: invalid input type: {type(x)}. Must be Tensor or PackedSequence")


        # lstm: (batch_size, seq_length, features) -> (batch_size, hidden_size)
        # or:   packed_sequence -> packed_sequence
        # out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out, (h_n, c_n) = self.lstm(x)  # (h0, c0) defaults to zeros

        # select the last state of lstm's neurons
        if type(out) == nn.utils.rnn.PackedSequence:
            # padding has to be considered
            out, lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # the unpadded length of batch i is lengths[i], so that is the last non-zero state
            out = torch.stack([out[i,lengths[i].item()-1,:] for i in range(len(lengths))])
        elif unpadded_lengths is not None:
            out = torch.stack([out[i,unpadded_lengths[i].item()-1,:] for i in range(len(unpadded_lengths))])
        else:
            if out.shape[0] == 3:  # batched
                out = out[:,-1,:]
            else:  # unbatched
                # softmax requires (batch_size, *)
                out = torch.stack([out[-1,:]])

        # fc fully connected layer: (*, hidden_size) -> (*, num_classes)
        out = self.fc(out)

        # softmax: (batch_size, *) -> (batch_size, *)
        out = self.softmax(out)
        return out
