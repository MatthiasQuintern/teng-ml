if __name__ == "__main__":
    if __package__ is None:
        # make relative imports work as described here: https://peps.python.org/pep-0366/#proposed-change
        __package__ = "teng-ml"
        import sys
        from os import path
        filepath = path.realpath(path.abspath(__file__))
        sys.path.insert(0, path.dirname(path.dirname(filepath)))

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from .util.transform import ConstantInterval
from .util.data_loader import load_datasets, LabelConverter

def test_interpol():
    file = "/home/matth/data/2023-04-27_glass_8.2V_179mm000.csv"
    # file = "/home/matth/data/test001.csv"
    df = pd.read_csv(file)
    array = df.to_numpy()
    print(ConstantInterval.get_average_interval(array[:,0]))
    transformer = ConstantInterval(0.05)
    interp_array = transformer(array[:,0], array[:,2])

    fig1, ax1 = plt.subplots()
    ax1.plot(interp_array[:,0], interp_array[:,1], color="r", label="Interpolated")
    ax1.scatter(array[:,0], array[:,2], color="g", label="Original")
    ax1.legend()
    plt.show()

if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    labels = LabelConverter(["foam", "glass", "kapton", "foil"])
    train_set, test_set = load_datasets("/home/matth/data", labels, voltage=8.2)

    # train_loader = iter(DataLoader(train_set))
    # test_loader = iter(DataLoader(test_set))
    # sample = next(train_loader)
    # print(sample)
    train_loader = iter(DataLoader(train_set))
    test_loader = iter(DataLoader(test_set))
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

    model = RNN(input_size = 1, hidden_size = 8, num_layers = 3, num_classes = 18, if_bidirectional = True).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    print(model)

# training 
    for ep in range(40):
        train_correct = 0
        train_total = 0
        val_correct = 0
        val_total = 0
        for (x, y), length in train_loader: 
            batch_size = x.shape[0]
            v = x.view(batch_size, -1, nFeatrue)
            data = rnn_utils.pack_padded_sequence(v.type(torch.FloatTensor), length, batch_first=True).to(device)
            # print(data.batch_sizes[0])
            # print(data)
            out = model(data)
            loss = loss_func(out, y) 
            # print(loss)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            
            predicted = torch.max(torch.nn.functional.softmax(out), 1)[1]
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()


        scheduler.step()
        
        for (x, y), length in test_loader: 
            batch_size = x.shape[0]
            v = x.view(batch_size, -1, nFeatrue)
            data = rnn_utils.pack_padded_sequence(v.type(torch.FloatTensor), length, batch_first=True).to(device)
            out = model(data)
            loss = loss_func(out, y)     
            
            predicted = torch.max(torch.nn.functional.softmax(out), 1)[1]
            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()

        print("epoch: ", ep + 1, 'Accuracy of the Train: %.2f %%' % (100 * train_correct / train_total), 'Accuracy of the Test: %.2f %%' % (100 * val_correct / val_total))

