if __name__ == "__main__":
    if __package__ is None:
        # make relative imports work as described here: https://peps.python.org/pep-0366/#proposed-change
        __package__ = "teng_ml"
        import sys
        from os import path
        filepath = path.realpath(path.abspath(__file__))
        sys.path.insert(0, path.dirname(path.dirname(filepath)))

from sys import exit
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import itertools
import time
from os import makedirs, path

from .util.transform import ConstantInterval, Normalize
from .util.data_loader import load_datasets, LabelConverter
from .util.split import DataSplitter
from .util.settings import MLSettings
from .rnn.rnn import RNN
from .rnn.training import train_validate_save, select_device

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
    # plt.show()


if __name__ == "__main__":
    labels = LabelConverter(["white_foam", "glass", "Kapton", "bubble_wrap", "cloth", "black_foam"])
    models_dir = "/home/matth/Uni/TENG/models"  # where to save models, settings and results
    if not path.isdir(models_dir):
        makedirs(models_dir)
    data_dir = "/home/matth/Uni/TENG/data"


    # Test with
    num_layers = [ 3 ]
    hidden_size = [ 8 ]
    bidirectional = [ True ]
    t_const_int = ConstantInterval(0.01)
    t_norm = Normalize(0, 1)
    transforms = [[ t_const_int ]] #, [ t_const_int, t_norm ]]
    batch_sizes = [ 64 ]  # , 16]
    splitters = [ DataSplitter(100) ]
    num_epochs = [ 80 ]

     # num_layers=1,
     # hidden_size=1,
     # bidirectional=True,
     # optimizer=None,
     # scheduler=None,
     # loss_func=None,
     # transforms=[],
     # splitter=None,
     # num_epochs=10,
     # batch_size=5,
    args = [num_layers, hidden_size, bidirectional, [None], [None], [None], transforms, splitters, num_epochs, batch_sizes]

    # create settings for every possible combination
    settings = [
        MLSettings(1, *params, labels) for params in itertools.product(*args)
    ]

    loss_func = nn.CrossEntropyLoss()
    optimizers = [
        lambda model: torch.optim.Adam(model.parameters(), lr=0.03),
        # lambda model: torch.optim.Adam(model.parameters(), lr=0.25),
        # lambda model: torch.optim.Adam(model.parameters(), lr=0.50),
    ]
    schedulers = [
        # lambda optimizer, st: torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
        lambda optimizer, st: torch.optim.lr_scheduler.StepLR(optimizer, step_size=st.num_epochs // 10, gamma=0.40, verbose=False),
        # lambda optimizer, st: torch.optim.lr_scheduler.StepLR(optimizer, step_size=st.num_epochs // 10, gamma=0.75, verbose=False),
    ]

    n_total = len(settings) * len(optimizers) * len(schedulers)
    print(f"Testing {n_total} possible configurations")
    # scheduler2 = 
    def create_model(st, optimizer_f, scheduler_f):
        model=RNN(input_size=st.num_features, hidden_size=st.hidden_size, num_layers=st.num_layers, num_classes=len(labels), bidirectional=st.bidirectional)
        optimizer = optimizer_f(model)
        scheduler = scheduler_f(optimizer, st)
        return model, optimizer, scheduler

    t_begin = time.time()
    n = 1
    for o in range(len(optimizers)):
        for s in range(len(schedulers)):
            for i in range(len(settings)):
                st = settings[i]
                # print(st.get_name())
                train_set, test_set = load_datasets(data_dir, labels, voltage=8.2, transforms=st.transforms, split_function=st.splitter, train_to_test_ratio=0.7, random_state=42, num_workers=4)

                generator = torch.manual_seed(42)
                # train_loader = iter(DataLoader(train_set))
                # test_loader = iter(DataLoader(test_set))
                train_loader = DataLoader(train_set, batch_size=st.batch_size, shuffle=True, generator=generator)
                test_loader = DataLoader(test_set, batch_size=st.batch_size, shuffle=True, generator=generator)
                print(f"Testing {n}/{n_total}: (o={o}, s={s}, i={i})")
                model, optimizer, scheduler = create_model(st, optimizers[o], schedulers[s])
                device = select_device(force_device="cpu")
                try:
                    train_validate_save(model, optimizer, scheduler, loss_func, train_loader, test_loader, st, models_dir, print_interval=1)
                except KeyboardInterrupt:
                    if input("Cancelled current training. Quit? (q/*): ") == "q":
                        t_end = time.time()
                        print(f"Testing took {t_end - t_begin:.2f}s = {(t_end-t_begin)/60:.1f}m")
                        exit()
                n += 1

    t_end = time.time()
    print(f"Testing took {t_end - t_begin:.2f}s = {(t_end-t_begin)/60:.1f}m")

