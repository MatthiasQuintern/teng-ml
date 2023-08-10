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
from .util.data_loader import load_datasets, LabelConverter, count_data
from .util.split import DataSplitter
from .util.pad import PadSequences
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
    # labels = LabelConverter(["foam_PDMS_white", "foam_PDMS_black", "foam_PDMS_TX100", "foam_PE", "antistatic_foil", "cardboard", "glass", "kapton", "bubble_wrap_PE", "fabric_PP", ])
    labels = LabelConverter(["foam_PDMS_white", "foam_PDMS_black", "foam_PDMS_TX100", "foam_PE", "kapton", "bubble_wrap_PE", "fabric_PP", ])
    models_dir = "/home/matth/Uni/TENG/teng_2/models_gen_8"  # where to save models, settings and results
    if not path.isdir(models_dir):
        makedirs(models_dir)
    data_dir = "/home/matth/Uni/TENG/teng_2/sorted_data"

    # gen_5 best options: datasplitter, not bidirectional, lr=0.001, no scheduler
    # gen_6 best options: no glass, cardboard and antistatic_foil, not bidirectional, lr=0.0007, no datasplitter, 2 layers n_hidden = 10

    # Test with
    num_layers = [ 2 ]
    hidden_size = [ 7, 11, 14 ]
    bidirectional = [ False, True ]
    t_const_int = ConstantInterval(0.01)  # TODO check if needed: data was taken at equal rate, but it isnt perfect -> maybe just ignore?
    t_norm = Normalize(-1, 1)
    transforms = [[ ], [ t_norm ]]  #, [ t_norm, t_const_int ]]
    batch_sizes = [ 4 ]
    splitters = [ DataSplitter(50, drop_if_smaller_than=30), DataSplitter(100, drop_if_smaller_than=30) ]  # smallest file has length 68 TODO: try with 0.5-1second snippets
    num_epochs = [ 5 ]
    # (epoch, min_accuracy)
    training_cancel_points = [(10, 10), (20, 20), (40, 30)]
    # training_cancel_points = []

    args = [num_layers, hidden_size, bidirectional, [None], [None], [None], transforms, splitters, num_epochs, batch_sizes]

    # create settings for every possible combination
    settings = [
        MLSettings(1, *params, labels) for params in itertools.product(*args)
    ]

    loss_func = nn.CrossEntropyLoss()
    optimizers = [
        lambda model: torch.optim.Adam(model.parameters(), lr=0.0005),
        lambda model: torch.optim.Adam(model.parameters(), lr=0.0007),
        # lambda model: torch.optim.Adam(model.parameters(), lr=0.008),
    ]
    schedulers = [
        None,
        # lambda optimizer, st: torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
        # lambda optimizer, st: torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5),
        lambda optimizer, st: torch.optim.lr_scheduler.StepLR(optimizer, step_size=st.num_epochs // 8, gamma=0.50, verbose=False),
        # lambda optimizer, st: torch.optim.lr_scheduler.StepLR(optimizer, step_size=st.num_epochs // 10, gamma=0.75, verbose=False),
    ]

    device = select_device(force_device="cpu")  # TODO cuda is not supported because something throws NotImplementedError with my gpu
    n_total = len(settings) * len(optimizers) * len(schedulers)
    print(f"Testing {n_total} possible configurations, device='{device}'")
    # scheduler2 = 
    def create_model(st, optimizer_f, scheduler_f):
        model=RNN(input_size=st.num_features, hidden_size=st.hidden_size, num_layers=st.num_layers, num_classes=len(labels), bidirectional=st.bidirectional)
        optimizer = optimizer_f(model)
        if scheduler_f is not None:
            scheduler = scheduler_f(optimizer, st)
        else: scheduler = None
        return model, optimizer, scheduler

    t_begin = time.time()
    n = 1
    for o in range(len(optimizers)):
        for s in range(len(schedulers)):
            for i in range(len(settings)):
                st = settings[i]
                train_set, test_set = load_datasets(data_dir, labels, exclude_n_object=None, voltage=None, transforms=st.transforms, split_function=st.splitter, train_to_test_ratio=0.7, random_state=80, num_workers=4)

                generator = torch.manual_seed(42)
                train_loader = DataLoader(train_set, batch_size=st.batch_size, shuffle=True, generator=generator, collate_fn=PadSequences())
                test_loader = DataLoader(test_set, batch_size=None, shuffle=True, generator=generator)

                # set batch_size to None and remove collate_fn for this to work
                # count_data(train_loader, st.labels, print_summary="training data")
                # count_data(test_loader, st.labels, print_summary="validation data")


                model, optimizer, scheduler = create_model(st, optimizers[o], schedulers[s])
                print(f"Testing {n}/{n_total}: (o={o}, s={s}, i={i})")
                try:
                    train_validate_save(model, optimizer, scheduler, loss_func, train_loader, test_loader, st, models_dir, print_interval=1, print_continuous=True, training_cancel_points=training_cancel_points)
                except KeyboardInterrupt:
                    if input("Cancelled current training. Quit? (q/*): ") == "q":
                        t_end = time.time()
                        print(f"Testing took {t_end - t_begin:.2f}s = {(t_end-t_begin)/60:.1f}m")
                        exit()
                n += 1

    t_end = time.time()
    print(f"Testing took {t_end - t_begin:.2f}s = {(t_end-t_begin)/60:.1f}m")

