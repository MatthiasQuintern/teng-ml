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

    def forward(self, x):
        device = x.device

        # h0: initial hidden states
        # c0: initial cell states
        if len(x.shape) == 2:  # x: (seq_length, features)
            h0 = torch.zeros(self.D * self.num_layers, self.hidden_size).to(device)
            c0 = torch.zeros(self.D * self.num_layers, self.hidden_size).to(device)
        elif len(x.shape) == 3:   # x: (batch, seq_length, features)
            batch_size = x.shape[0]
            h0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.D * self.num_layers, batch_size, self.hidden_size).to(device)
        else:
            raise ValueError(f"RNN.forward: invalid iput shape: {x.shape}. Must be (batch, seq_length, features) or (seq_length, features)")

        # lstm: (batch_size, seq_length, features) -> (batch_size, hidden_size)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        print(f"forward: out.shape={out.shape} TODO verify comment")
        # out: (N, L, D * hidden_size)
        # h_n: (D * num_layers, hidden_size)
        # c_n: (D * num_layers, hidden_size)
        # print(f"out({out.shape})={out}")
        # print(f"h_n({h_n.shape})={h_n}")
        # print(f"c_n({c_n.shape})={c_n}")
        # print(f"out({out.shape})=...")
        # print(f"h_n({h_n.shape})=...")
        # print(f"c_n({c_n.shape})=...")

        """
        # select only last layer [-1] -> last layer,
        last_layer_state  = h_n.view(self.num_layers, D, batch_size, self.hidden_size)[-1]
        if D == 1:
            # [1, batch_size, hidden_size] -> [batch_size, hidden_size]
            X = last_layer_state.squeeze()           # TODO what if batch_size == 1
        elif D == 2:
            h_1, h_2 = last_layer_state[0], last_layer_state[1]   # states of both directions
            # concatenate both states, X-size: (Batch, hidden_size * 2）
            X = torch.cat((h_1, h_2), dim=1)
        else:
            raise ValueError("D must be 1 or 2")
        """  # all this is quivalent to line below
        out = out[:,-1,:]  # select last time step

        # fc fully connected layer: (*, hidden_size) -> (*, num_classes)
        out = self.fc(out)

        # softmax: (*) -> (*)
        out = self.softmax(out)
        return out
