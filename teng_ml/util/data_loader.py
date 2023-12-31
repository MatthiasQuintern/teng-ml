
from os import path, listdir
import re
import numpy as np
import pandas as pd
from scipy.sparse import data
import torch

import threading

from sklearn.model_selection import train_test_split

from teng_ml.util.transform import Multiply

# groups: date, name, n_object, voltage, distance, index
# re_filename = r"(\d{4}-\d{2}-\d{2})_([a-zA-Z_]+)_(\d{1,2}(?:\.\d*)?)V_(\d+(?:\.\d*)?)mm(\d+).csv"
# for teng_1
# re_filename = r"(\d{4}-\d{2}-\d{2})_([a-zA-Z_]+)_()(\d{1,2}(?:\.\d*)?)V_(\d+(?:\.\d*)?)mm(\d+).csv"
# for teng_2
re_filename = r"(\d{4}-\d{2}-\d{2})_([a-zA-Z0-9_]+)_(\d+)_(\d{1,2}(?:\.\d*)?)V_(\d+(?:\.\d*)?)mm(\d+).csv"

class LabelConverter:
    def __init__(self, class_labels: list[str]):
        self.class_labels = class_labels.copy()
        self.class_labels.sort()

    def get_one_hot(self, label):
        """return one hot vector for given label"""
        vec = np.zeros(len(self.class_labels), dtype=np.float32)
        vec[self.class_labels.index(label)] = 1.0
        return vec

    def get_label_index(self, one_hot: torch.Tensor):
        """return one hot vector for given label"""
        return int(torch.argmax(one_hot).item())

    def __getitem__(self, index):
        if type(index) == torch.Tensor:
            return self.class_labels[self.get_label_index(index)]
        return self.class_labels[index]

    def __contains__(self, value):
        return value in self.class_labels

    def __len__(self):
        return len(self.class_labels)

    def get_labels(self):
        return self.class_labels.copy()

    def __repr__(self):
        return str(self.class_labels)


class Datasample:
    def __init__(self, date: str, label: str, n_object: str, voltage: str, distance: str, index: str, label_vec, datapath: str, init_data=False):
        self.date = date
        self.label = label
        self.n_object = 0 if n_object == "" else int(n_object)
        self.voltage = float(voltage)
        self.distance = float(distance)
        self.index = int(index)
        self.label_vec = label_vec
        self.datapath = datapath
        self.data = None
        if init_data: self._load_data()

    def __repr__(self):
        size = self.data.size if self.data is not None else "Unknown"
        return f"{self.label}-{self.index}: dimension={size}, recorded at {self.date} with U={self.voltage}V, d={self.distance}mm"

    def _load_data(self):
        # df = pd.read_csv(self.datapath)
        self.data = np.loadtxt(self.datapath, skiprows=1, dtype=np.float32, delimiter=",")

    def get_data(self):
        """[[timestamp, idata, vdata]]"""
        if self.data is None:
            self._load_data()
        return self.data


class Dataset:
    """
    Store the whole dataset, compatible with torch.data.Dataloader
    """
    def __init__(self, datasamples, transforms=[], split_function=None):
        """
        @param transforms: single callable or list of callables that are applied to the data (before eventual split)
        @param split_function: (data) -> [data0, data1...] callable that splits the data
        """
        self.transforms = transforms
        self.data = []  # (data, label)

        # NORMALIZE ALL DATA WITH THE SAME FACTOR
        # sup = 0
        # inf = 0
        # for sample in datasamples:
        #     data = sample.get_data()
        #     max_ = np.max(data[:,2])
        #     min_ = np.min(data[:,2])
        #     if max_ > sup: sup = max_
        #     if min_ < inf: inf = min_
        # multiplier = 1 / max(sup, abs(inf))
        # self.transforms.append(Multiply(multiplier))

        for sample in datasamples:
            data = self.apply_transforms(sample.get_data())
            if split_function is None:
                self.data.append((data, sample.label_vec))
            else:
                try:
                    for data_split in split_function(data):
                        self.data.append((data_split, sample.label_vec))
                except ValueError as e:
                    raise ValueError(f"Exception occured during splitting of sample '{sample.datapath}': {e}")

    def apply_transforms(self, data):
        if type(self.transforms) == list:
            for t in self.transforms:
                data = t(data)
        elif self.transforms is not None:
            data = self.transforms(data)
        return data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_datafiles(datadir, labels: LabelConverter, exclude_n_object=None, filter_voltage=None):
    """
    get a list of all matching datafiles from datadir that are in the format: yyyy-mm-dd_label__n_object_x.xV_xxxmm.csv
    """
    datafiles = []
    files = listdir(datadir)
    files.sort()
    for file in files:
        match = re.fullmatch(re_filename, file)
        if not match:
            print(f"get_datafiles: dropping non matching file '{file}'")
            continue

        label = match.groups()[1]
        if label not in labels: continue

        sample_n_object = 0 if match.groups()[2] == "" else int(match.groups()[2])
        if exclude_n_object and exclude_n_object == sample_n_object: continue
        sample_voltage = float(match.groups()[3])
        if filter_voltage and filter_voltage != sample_voltage: continue

        datafiles.append((datadir + "/" + file, match, label))
    return datafiles


def load_datasets(datadir, labels: LabelConverter, transforms=None, split_function=None, exclude_n_object=None, voltage=None, train_to_test_ratio=0.7, random_state=None, num_workers=None):
    """
    load all data from datadir that are in the format: yyyy-mm-dd_label_x.xV_xxxmm.csv
    """
    datasamples = []
    if num_workers == None:
        for file, match, label in get_datafiles(datadir, labels, exclude_n_object=exclude_n_object, filter_voltage=voltage):
            datasamples.append(Datasample(*match.groups(), labels.get_one_hot(label), file))
    else:
        files = get_datafiles(datadir, labels, exclude_n_object=exclude_n_object, filter_voltage=voltage)
        def worker():
            while True:
                try:
                    file, match, label = files.pop()
                except IndexError:
                    # No more files to process
                    return
                datasamples.append(Datasample(*match.groups(), labels.get_one_hot(label), file, init_data=True))
        threads = [threading.Thread(target=worker) for _ in range(num_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # TODO do the train_test_split after the Dataset split
    # problem: needs to be after transforms
    train_samples, test_samples = train_test_split(datasamples, train_size=train_to_test_ratio, shuffle=True, random_state=random_state)
    train_dataset = Dataset(train_samples, transforms=transforms, split_function=split_function)
    test_dataset = Dataset(test_samples, transforms=transforms, split_function=split_function)
    return train_dataset, test_dataset


def count_data(data_loader, label_converter: LabelConverter, print_summary=False):
    """
    @param data_loader: unbatched data loader
    """
    n_sequences = 0  # count number of sequences
    labels = [ 0 for _ in range(len(label_converter)) ]     # count number of sequences per label
    len_data = [ 0 for _ in range(len(label_converter)) ]   # count number of datapoints per label
    for i, (data, y) in enumerate(data_loader):
        n_sequences = i
        label_i = label_converter.get_label_index(y)
        len_data[label_i] += data.shape[0]
        labels[label_i] += 1
    if print_summary:
        print("=" * 50)
        print("Dataset summary" + f" for {print_summary}:" if type(print_summary) == str else ":")
        print(f"Number of sequences: {n_sequences}")
        for i in range(len(label_converter)):
            print(f"- {label_converter[i]:15}: {labels[i]:3} sequences, {len_data[i]:5} datapoints")

    return n_sequences, labels, len_data





