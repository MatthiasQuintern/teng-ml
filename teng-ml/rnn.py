import torch
import torch.nn as nn

# BiLSTM Model

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes, if_bidirectional):
    super(RNN, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.if_bidirectional = if_bidirectional
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=if_bidirectional)

    if if_bidirectional == True:
      self.fc = nn.Linear(hidden_size * 2, num_classes)
    else:
      self.fc = nn.Linear(hidden_size, num_classes)


  def forward(self, x):
    D = 2 if self.if_bidirectional == True else 1
    Batch = x.batch_sizes[0]

    h0 = torch.zeros(D * self.num_layers, Batch, self.hidden_size).to(device)
    c0 = torch.zeros(D * self.num_layers, Batch, self.hidden_size).to(device)
    x.to(device)
    _, (h_n, _) = self.lstm(x, (h0, c0))
    final_state  = h_n.view(self.num_layers, D, Batch, self.hidden_size)[-1]     # num_layers, num_directions, batch, hidden_size

    if D == 1:
      X = final_state.squeeze()
    elif D == 2:
      h_1, h_2 = final_state[0], final_state[1]  # forward & backward pass
      #X = h_1 + h_2                # Add both states
      X = torch.cat((h_1, h_2), 1)         # Concatenate both states, X-size: (Batch, hidden_size * 2）

    output = self.fc(X) # fully-connected layer

    return output
