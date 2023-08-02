import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from time import sleep
from random import choice as r_choice
from sys import exit
import os
import re


if __name__ == "__main__":
    if __package__ is None:
        # make relative imports work as described here: https://peps.python.org/pep-0366/#proposed-change
        __package__ = "teng_ml"
        import sys
        from os import path
        filepath = path.realpath(path.abspath(__file__))
        sys.path.insert(0, path.dirname(path.dirname(filepath)))

from .util.transform import Normalize
from .util.data_loader import get_datafiles
from .util.file_io import get_next_digits

file = "/home/matth/Uni/TENG/teng_2/data/2023-06-28_foam_black_1_188mm_06V001.csv"


class InteractiveDataSelector:
    re_file = r'\d{4}-\d{2}-\d{2}_([a-zA-Z_]+)_([a-zA-Z0-9]+)_(\d+(?:\.\d+)?mm)_(\d+V)(\d+)\.csv'
    re_index_group_nr = 5  # group number of the index part of the filename
    """
    Go through all .csv files in a directory, split the data and exclude sections with the mouse, then write the sections as single files into a new directory
    """
    def __init__(self, in_dir, out_dir, keep_index=True, split_at_exclude=True):
        """
        @param keep_index:
            If True: append the split number as triple digits to the existing filename (file001.csv -> file001001.csv, file001002.csv ...)
            Else: remove the indices from the filename before adding the split number (file001.csv -> file001.csv, file002.csv ...)
        @param split_at_exclude:
            If True: When excluding an area, split the data before and after the excluded zone
            Else: remove the excluded zone and join the previous and later part
        """
        if os.path.isdir(out_dir):
            if os.listdir(out_dir):
                raise ValueError(f"'out_dir' = '{out_dir}' is not empty")
        else:
            os.makedirs(out_dir)
        self._out_dir = out_dir

        self._in_dir = in_dir
        self._in_files = os.listdir(in_dir)
        self._in_files.sort()
        for i in reversed(range(len(self._in_files))):
            if not re.fullmatch(InteractiveDataSelector.re_file, self._in_files[i]):
                print(f"Dropping non-matching file '{self._in_files[i]}'")
                self._in_files.pop(i)
        if not self._in_files:
            raise ValueError(f"No matching files in 'in_dir' = '{in_dir}'")

        self._keep_index = keep_index
        self.split_at_exclude = split_at_exclude

        plt.ion()
        self._fig, self._ax = plt.subplots()

        self._fig.canvas.mpl_connect("button_press_event", lambda ev: self._fig_on_button_press(ev))
        self._fig.canvas.mpl_connect("key_press_event", lambda ev: self._fig_on_key_press(ev))

    def run(self):
        self._next_file()
        while plt.fignum_exists(self._fig.number):
            plt.pause(0.01)

    def _set_titles(self):
        help_str = "[(e)xclude, (S)plit, (w)rite]"
        self._fig.suptitle(f"{help_str}\ncurret mode: {self._mode}")

    def _next_file(self):
        # runtime stuff
        if len(self._in_files) == 0:
            raise IndexError("No more files to process")
        self._current_file = self._in_files.pop(0)
        self._current_dataframe = pd.read_csv(os.path.join(self._in_dir, self._current_file))
        self._current_array = self._current_dataframe.to_numpy()
        # self._current_array = np.loadtxt(os.path.join(self._in_dir, self._current_file), skiprows=1, delimiter=",")


        # plot stuff
        self._splits_lines = None  # vlines
        self._excludes_lines = None
        self._excludes_areas = [] # list of areas
        self._fig.clear()
        self._ax = self._fig.subplots()
        self._ax.plot(self._current_array[:,0], self._current_array[:,2])
        self._ax.set_xlabel(self._current_file)

        self._splits: list[int] = []
        self._excludes: list[int] = []
        self._mode = "exclude" # split or exclude
        self._set_titles()

        self._set_titles()

    def _fig_on_button_press(self, event):
        """
        left click: set split / exclude section (depends on mode)
        right click: undo last action of selected mode
        """
        if event.xdata is None: return
        if event.xdata in self._excludes or event.xdata in self._splits: return
        if event.button == 1:  # left click, add position
            if self._mode == "split":
                self._splits.append(event.xdata)
            else:
                self._excludes.append(event.xdata)
        elif event.button == 3:  # right click, undo
            if self._mode == "split":
                if len(self._splits) > 0:
                    self._splits.pop()
            else:
                if len(self._excludes) > 0:
                    self._excludes.pop()
        self._update_lines()

    def _fig_on_key_press(self, event):
        """
        S: set split mode
        e: set exclude mode
        w: write and got to next file
        """
        if event.key == 'S':
            self._mode = "split"
        elif event.key == 'e':
            self._mode = "exclude"
        elif event.key == 'w':
            self._save_as_new_files()
        self._set_titles()

    def _update_lines(self):
        # print(self._splits, self._excludes)
        ymin, ymax = self._ax.get_ylim()

        if self._splits_lines is not None: self._splits_lines.remove()
        self._splits_lines = self._ax.vlines(self._splits, ymin, ymax, color="b")

        if self._excludes_lines is not None: self._excludes_lines.remove()
        self._excludes_lines = self._ax.vlines(self._excludes, ymin, ymax, color="r")

        for area in self._excludes_areas:
            area.remove()
        self._excludes_areas.clear()
        excludes = self._excludes.copy()
        if len(excludes) % 2 == 1: excludes.pop()  # only draw pairs
        excludes.sort()
        for i in range(1, len(excludes), 2):
            self._excludes_areas.append(self._ax.axvspan(excludes[i-1], excludes[i], facecolor='r', alpha=0.3))

        self._ax.set_ylim(ymin, ymax)  # reset, since margins are added to lines
        self._fig.canvas.draw()

    def _get_next_filename(self):
        if self._keep_index:
            # 5th group is index
            match = re.fullmatch(InteractiveDataSelector.re_file, self._current_file)
            assert(type(match) is not None)
            basename = self._current_file[:match.start(InteractiveDataSelector.re_index_group_nr)]
        else:
            basename = self._current_file[:-4]  # extension
        index = get_next_digits(basename, self._out_dir, digits=3)
        return f"{basename}{index}.csv"

    def _save_as_new_files(self):
        # convert timestamps to their closest index
        excludes_idx = [np.abs(self._current_array[:,0] - t).argmin() for t in self._excludes]
        splits_idx = [np.abs(self._current_array[:,0] - t).argmin() for t in self._splits]
        if self.split_at_exclude:
            # split before the start of the exclucded range
            splits_idx += [ excludes_idx[i]-1 for i in range(0, len(excludes_idx), 2) ]
            # split after the end of the exclucded range
            splits_idx += [ excludes_idx[i]+1 for i in range(1, len(excludes_idx), 2) ]
        splits_idx = list(set(splits_idx))  # remove duplicates
        splits_idx.sort()

        df = self._current_dataframe.copy()

        # 1) remove excluded parts
        for i in range(1, len(excludes_idx), 2):
            df = df.drop(index=range(excludes_idx[i-1], excludes_idx[i]+1))

        # 2) splits
        new_frames = []
        start_i = df.index[0]
        for i in range(0, len(splits_idx)):
            end_i = splits_idx[i]
            # print(start_i, end_i)
            # check if valid start and end index
            if start_i in df.index and end_i in df.index:
                new_frames.append(df.loc[start_i:end_i])
            start_i = end_i + 1
        # append rest
        if start_i in df.index:
            new_frames.append(df.loc[start_i:])

        # 3) remove empty
        for i in reversed(range(len(new_frames))):
            if len(new_frames[i]) == 0:
                new_frames.pop(i)

        for frame in new_frames:
            filename = self._get_next_filename()
            pathname = os.path.join(self._out_dir, filename)
            # until now, frame is a copy of a slice
            frame = frame.copy()
            # transform timestamps so that first value is 0
            t_column_name = frame.columns[0]
            frame[t_column_name] -= frame.iloc[0][t_column_name]
            frame.to_csv(pathname, index=False)
            print(f"Saved range of length {len(frame.index):04} to {pathname}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("data_preprocess")
    parser.add_argument("in_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--keep_index", action="store_true")
    ns = parser.parse_args()

    selector = InteractiveDataSelector(ns.in_dir, ns.out_dir, ns.keep_index)
    selector.run()

