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
import json
import time
import pickle

from .util.transform import ConstantInterval, Normalize
from .util.data_loader import load_datasets, LabelConverter
from .util.epoch_tracker import EpochTracker
from .util.settings import MLSettings

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


    labels = LabelConverter(["foam", "glass", "kapton", "foil", "cloth", "rigid_foam"])
    t_const_int = ConstantInterval(0.01)
    t_norm = Normalize(0, 1)
    transforms = [ t_const_int, t_norm ]
    st = MLSettings(num_features=1,
                    num_layers=1,
                    hidden_size=1,
                    bidirectional=True,
                    transforms=transforms,
                    num_epochs=40,
                    batch_size=3,
                    labels=labels,
                )

    print(f"Using device: {device}")


    train_set, test_set = load_datasets("/home/matth/Uni/TENG/data", labels, voltage=8.2, transforms=st.transforms, train_to_test_ratio=0.7, random_state=42)

    # train_loader = iter(DataLoader(train_set))
    # test_loader = iter(DataLoader(test_set))
    train_loader = DataLoader(train_set, batch_size=st.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=st.batch_size, shuffle=True)

    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional):
            super(RNN, self).__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.is_bidirectional = bidirectional
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
            # x = (batch_size, sequence, feature)

            if bidirectional == True:
              self.fc = nn.Linear(hidden_size * 2, num_classes)
            else:
              self.fc = nn.Linear(hidden_size, num_classes)

            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            # x: batches, length, features
            # print(f"forward pass")
            D = 2 if self.is_bidirectional == True else 1

            # print(f"x({x.shape})=...")
            batch_size = x.shape[0]
            # print(f"batch_size={batch_size}")

            h0 = torch.zeros(D * self.num_layers, batch_size, self.hidden_size).to(device)
            # print(f"h1({h0.shape})=...")
            c0 = torch.zeros(D * self.num_layers, batch_size, self.hidden_size).to(device)
            x.to(device)
            _, (h_n, _) = self.lstm(x, (h0, c0))
            # print(f"h_n({h_n.shape})=...")
            final_state  = h_n.view(self.num_layers, D, batch_size, self.hidden_size)[-1]     # num_layers, num_directions, batch, hidden_size
            # print(f"final_state({final_state.shape})=...")

            if D == 1:
                X = final_state.squeeze()  # TODO what if batch_size == 1
            elif D == 2:
              h_1, h_2 = final_state[0], final_state[1]  # forward & backward pass
              #X = h_1 + h_2                        # Add both states
              X = torch.cat((h_1, h_2), 1)          # Concatenate both states, X-size: (Batch, hidden_size * 2）
            else:
                raise ValueError("D must be 1 or 2")
            # print(f"X({X.shape})={X}")
            output = self.fc(X) # fully-connected layer
            # print(f"out({output.shape})={output}")
            output = self.softmax(output)
            # print(f"out({output.shape})={output}")
            return output

    model=RNN(input_size=st.num_features, hidden_size=st.hidden_size, num_layers=st.num_layers, num_classes=len(labels), bidirectional=st.bidirectional).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    print(f"model:", model)
    print(f"loss_func={loss_func}")
    print(f"optimizer={optimizer}")
    print(f"scheduler={scheduler}")



    epoch_tracker = EpochTracker(labels)

    print(f"train_loader")
    for i, (data, y) in enumerate(train_loader):
        print(y)
        print(f"{i:3} - {torch.argmax(y, dim=1, keepdim=False)}")


# training 
    epoch_tracker.train_begin()
    for ep in range(st.num_epochs):
        for i, (data, y) in enumerate(train_loader):
            # print(data, y)
            # data = batch, seq, features
            # print(f"data({data.shape})={data}")
            x = data[:,:,[2]].float()   # select voltage data
            # print(f"x({x.shape}, {x.dtype})=...")
            # print(f"y({y.shape}, {y.dtype})=...")
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
            # print(f"out({out.shape}={out})")
            loss = loss_func(out, y)
            # print(loss)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            # predicted = torch.max(torch.nn.functional.softmax(out), 1)[1]
            predicted = torch.argmax(out, dim=1, keepdim=False)  # -> [ label_indices ]
            correct = torch.argmax(y, dim=1, keepdim=False)  # -> [ label_indices ]
            # print(f"predicted={predicted}, correct={correct}")
            # train_total += y.size(0)
            # train_correct += (predicted == correct).sum().item()
            epoch_tracker.train(correct, predicted)
        epoch_tracker.next_epoch(loss)
        print(epoch_tracker.get_last_epoch_summary_str())
        scheduler.step()
    t_end = time.time()

    with torch.no_grad():
        for i, (data, y) in enumerate(test_loader):
            # print(ep, "Test")
            x = data[:,:,[2]].float()
            out = model(x)
            loss = loss_func(out, y)

            predicted = torch.argmax(out, dim=1, keepdim=False)  # -> [ label_indices ]
            correct = torch.argmax(y, dim=1, keepdim=False)  # -> [ label_indices ]
            # print(f"predicted={predicted}, correct={correct}")
            # val_total += y.size(0)
            # val_correct += (predicted == correct).sum().item()

            epoch_tracker.test(correct, predicted)

        # print(f"train_total={train_total}, val_total={val_total}")
        # if train_total == 0: train_total = -1
        # if val_total == 0: val_total = -1

        # print(f"epoch={ep+1:3}: Testing accuracy={100 * val_correct / val_total:.2f}")
    # print(f"End result: Training accuracy={100 * train_correct / train_total:.2f}%, Testing accuracy={100 * val_correct / val_total:.2f}, training took {t_end - t_begin:.2f} seconds")

    epoch_tracker.get_test_statistics()
    # epoch_tracker.()

    # print(epoch_tracker.get_training_summary_str())
    print(epoch_tracker.get_training_count_per_label())

    model_name = st.get_name()
    # save the settings, results and model
    with open(model_name + "_settings.pkl", "wb") as file:
        pickle.dump(st, file)

    with open(model_name + "_results.pkl", "wb") as file:
        pickle.dump(epoch_tracker, file)

    with open(model_name + "_model.pkl", "wb") as file:
        pickle.dump(model, file)

