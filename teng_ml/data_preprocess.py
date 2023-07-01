import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from time import sleep
from random import choice as r_choice
from sys import exit


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

file = "/home/matth/Uni/TENG/teng_2/data/2023-06-28_foam_black_1_188mm_06V001.csv"

class InteractiveDataSelector:
    """
    Helper class for "iterating" through selected peaks.
    """
    def __init__(self, out_name, out_dir, fig, ax):
        self._out_dir = out_dir
        self._out_name = out_name
        self._fig = fig
        self._ax = ax

        self._fig.canvas.mpl_connect("button_press_event", lambda ev: self._fig_on_button_press(ev))
        self._fig.canvas.mpl_connect("key_press_event", lambda ev: self._fig_on_key_press(ev))

        self._splits_lines = None  # vlines
        self._excludes_lines = None
        self._excludes_areas = [] # list of areas

        self._splits: list[int] = []
        self._excludes: list[int] = []
        self._mode = None  # split or exclude
        self._set_mode("split")

    def run(self):
        while plt.fignum_exists(self._fig.number):
            plt.pause(0.01)

    def _fig_on_button_press(self, event):
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
        if event.key == 'S':
            self._set_mode("split")
        elif event.key == 'e':
            self._set_mode("exclude")

    def _set_mode(self, mode):
        help_str = "[(e)xclude - (S)plit]"
        if mode == "split":
            self._mode = "split"
            fig.suptitle(f"-> split mode {help_str}")
        else:
            self._mode = "exclude"
            fig.suptitle(f"-> exclude mode {help_str}")

    def _update_lines(self):
        print(self._splits, self._excludes)
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

    def _save_as_new_files(self):




if __name__ == "__main__":
    """
    Peak identification:
    plot, let user choose first, second, last and lowest peak for identification
    """
    df = pd.read_csv(file)
    a = df.to_numpy()

    # a2 = interpolate_to_linear_time()
    # print(a2)
    # exit()

    vdata = Normalize(0, 1)(a[:,2])
    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(vdata)
    ax.grid(True)
    selector = InteractiveDataSelector("bla", "test", fig, ax)
    selector.run()

