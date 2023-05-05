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
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

from .util.transform import ConstantInterval, Normalize
from .util.data_loader import load_datasets, LabelConverter

def test_interpol():
    file = "/home/matth/data/2023-04-27_glass_8.2V_179mm000.csv"
    # file = "/home/matth/data/test001.csv"
    df = pd.read_csv(file)
    array = df.to_numpy()
    print(ConstantInterval.get_average_interval(array[:,0]))
    transformer = ConstantInterval(0.05)
    interp_array = transformer(array[:,[0,2]])

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

    labels = LabelConverter(["foam", "glass", "kapton", "foil", "cloth", "rigid_foam"])
    t_const_int = ConstantInterval(0.01)
    t_norm = Normalize(0, 1)
    train_set, test_set = load_datasets("/home/matth/Uni/TENG/testdata", labels, voltage=8.2, transforms=[t_const_int], train_to_test_ratio=0.7, random_state=42)

    # train_loader = iter(DataLoader(train_set))
    # test_loader = iter(DataLoader(test_set))
    train_loader = iter(DataLoader(train_set, batch_size=3, shuffle=True))
    test_loader = iter(DataLoader(test_set, batch_size=3, shuffle=True))
# , dtype=torch.float32
    sample = next(train_loader)
    print(sample)

    feature_count = 1


    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, if_bidirectional):
            super(RNN, self).__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.if_bidirectional = if_bidirectional
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=if_bidirectional)
            # x = (batch_size, sequence, feature)

            if if_bidirectional == True:
              self.fc = nn.Linear(hidden_size * 2, num_classes)
            else:
              self.fc = nn.Linear(hidden_size, num_classes)


        def forward(self, x):
            # x: batches, length, features
            print(f"forward pass")
            D = 2 if self.if_bidirectional == True else 1

            print(f"x({x.shape})=...")
            batch_size = x.shape[0]
            print(f"batch_size={batch_size}")

            h0 = torch.zeros(D * self.num_layers, batch_size, self.hidden_size).to(device)
            print(f"h0({h0.shape})=...")
            c0 = torch.zeros(D * self.num_layers, batch_size, self.hidden_size).to(device)
            x.to(device)
            _, (h_n, _) = self.lstm(x, (h0, c0))
            print(f"h_n({h_n.shape})=...")
            final_state  = h_n.view(self.num_layers, D, batch_size, self.hidden_size)[-1]     # num_layers, num_directions, batch, hidden_size
            print(f"final_state({final_state.shape})=...")

            if D == 1:
              X = final_state.squeeze()
            elif D == 2:
              h_1, h_2 = final_state[0], final_state[1]  # forward & backward pass
              #X = h_1 + h_2                # Add both states
              X = torch.cat((h_1, h_2), 1)         # Concatenate both states, X-size: (Batch, hidden_size * 2）
            else:
                raise ValueError("D must be 1 or 2")
            output = self.fc(X) # fully-connected layer
            print(f"out({output.shape})={output}")

            return output

    model=RNN(input_size=1, hidden_size=8, num_layers=3, num_classes=18, if_bidirectional=True).to(device)
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
        for data, y in train_loader:
            # data = batch, seq, features
            print(ep, "Train")
            # print(f"data({data.shape})={data}")
            x = data[:,:,[2]].float()   # select voltage data
            print(f"x({x.shape}, {x.dtype})=...")
            print(f"y({y.shape}, {y.dtype})=...")
            # length = torch.tensor([x.shape[1] for _ in range(x.shape[0])], dtype=torch.int64)
            # print(f"length({length.shape})={length}")
            # batch_size = x.shape[0]
            # print(f"batch_size={batch_size}")
            # v = x.view(batch_size, -1, feature_count)
            # data = rnn_utils.pack_padded_sequence(v.type(torch.FloatTensor), length, batch_first=True).to(device)[0]
            # print(f"data({data.shape})={data}")
            # print(data.batch_sizes[0])
            # print(data)
            out = model(x)
            loss = loss_func(out, y)
            # print(loss)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            predicted = torch.max(torch.nn.functional.softmax(out), 1)[1]
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        scheduler.step()

        for data, y in test_loader:
            print(ep, "Test")
            x = data[:,:,[2]]
            print(f"x({x.shape})={x}")
            # length = torch.tensor(x.shape[1], dtype=torch.int64)
            # print(f"length={length}")
            # batch_size = x.shape[0]
            # print(f"batch_size={batch_size}")
            # v = x.view(batch_size, -1, feature_count)
            # data = rnn_utils.pack_padded_sequence(v.type(torch.FloatTensor), length, batch_first=True).to(device)
            out = model(x)
            loss = loss_func(out, y)

            predicted = torch.max(torch.nn.functional.softmax(out), 1)[1]
            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()

        print("epoch: ", ep + 1, 'Accuracy of the Train: %.2f %%' % (100 * train_correct / train_total), 'Accuracy of the Test: %.2f %%' % (100 * val_correct / val_total))

