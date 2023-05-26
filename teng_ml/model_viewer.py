from os import path, listdir

import matplotlib.pyplot as plt
import numpy as np
from sys import exit, argv

if __name__ == "__main__":
    if __package__ is None:
        # make relative imports work as described here: https://peps.python.org/pep-0366/#proposed-change
        __package__ = "teng_ml"
        import sys
        from os import path
        filepath = path.realpath(path.abspath(__file__))
        sys.path.insert(0, path.dirname(path.dirname(filepath)))

from .tracker.epoch_tracker import EpochTracker
from .util.settings import MLSettings
from .util import model_io as mio
from .util.string import cleanup_str, fill_and_center

cache = {}

#
# TOOLS
#
def get_model_dirs(models_dir):
    """
    return all model_dirs, relative to models_dir
    """
    if "model_dirs" in cache: return cache["model_dirs"].copy()
    paths = listdir(models_dir)
    model_dirs = []
    for p in paths:
        if not path.isdir(f"{models_dir}/{p}"): continue
        if not path.isfile(f"{models_dir}/{p}/settings.pkl"): continue
        if not path.isfile(f"{models_dir}/{p}/tracker_training.pkl"): continue
        if not path.isfile(f"{models_dir}/{p}/tracker_validation.pkl"): continue
        if not path.isfile(f"{models_dir}/{p}/model.pkl"): continue
        model_dirs.append(f"{models_dir}/{p}")
    cache["model_dirs"] = model_dirs.copy()
    return model_dirs


def resave_images_svg(model_dirs):
    """
    open all trackers and save all plots as svg
    """
    for model_dir in model_dirs:
        val_tracker: EpochTracker = mio.load_tracker_validation(model_dir)
        fig, _ = val_tracker.plot_predictions("Validation: Predictions", model_dir=model_dir, name="img_validation_predictions")
        train_tracker: EpochTracker = mio.load_tracker_training(model_dir)
        fig, _ = train_tracker.plot_predictions("Training: Predictions", model_dir=model_dir, name="img_training_predictions")
        fig, _ = train_tracker.plot_training(model_dir=model_dir)
        plt.close('all')

#
# MODEL RANKING
#
def get_model_info_md(model_dir):
    st: MLSettings = mio.load_settings(model_dir)
    validation_tracker = mio.load_tracker_validation(model_dir)
    training_tracker = mio.load_tracker_training(model_dir)

    s = f"""Model {model_dir[model_dir.rfind('/')+1:]}
Model parameters:
- num_features      = {st.num_features}
- num_layers        = {st.num_layers}
- hidden_size       = {st.hidden_size}
- bidirectional     = {st.bidirectional}
Training data:
- transforms        = {st.transforms}
- splitter          = {st.splitter}
- labels            = {st.labels}
Training info:
- optimizer         = {cleanup_str(st.optimizer)}
- scheduler         = {cleanup_str(st.scheduler)}
- loss_func         = {st.loss_func}
- num_epochs        = {st.num_epochs}
- batch_size        = {st.batch_size}
- n_predictions     = {np.sum(training_tracker.get_count_per_label())}
- final accuracy    = {training_tracker.accuracies[-1]}
- highest accuracy  = {np.max(training_tracker.accuracies)}
Validation info:
- n_predictions     = {np.sum(validation_tracker.get_count_per_label())}
- accuracy          = {validation_tracker.accuracies[-1]}
"""
    return s


def write_model_info(model_dir, model_info=None):
    if model_info is None: model_info = get_model_info_md(model_dir)
    with open(f"{model_dir}/model_info.md", "w") as file:
        file.write(model_info)


def get_model_ranking(model_dirs):
    if "model_ranking" in cache: return cache["model_ranking"].copy()
    model_ranking = []  # model, (model_dir | validation accuracy)
    for model_dir in model_dirs:
        model_ranking.append((model_dir, mio.load_tracker_validation(model_dir).accuracies[-1]))
    model_ranking.sort(key=lambda t: t[1])   # sort accuracy
    model_ranking.reverse()  # best to worst
    cache["model_ranking"] = model_ranking.copy()
    return model_ranking

def get_model_ranking_md(model_dirs):
    model_ranking = get_model_ranking(model_dirs)
    ranking_md = ""
    for i in range(len(model_ranking)):
        model_dir = model_ranking[i][0]
        model_name = model_dir[model_dir.rfind("/")+1:]
        ranking_md += f"{i+1:3}. Model=`{model_name}`, Validation accuaracy={round(model_ranking[i][1], 2):.2f}%\n"
    return ranking_md

#
# SETTINGS RANKING
#
def get_settings_ranking(model_dirs, use_ranking_instead_of_accuracy=False):
    """
    load the settings for each model and score them based on the performance of the model
    This only works when all settings were the same number of times
    (Example: 2 batch sizes x and y have to both to be used z times for the ranking to make sense)
    """
    if "settings_ranking" in cache: return cache["settings_ranking"].copy()
    settings_ranking = {}  # parameter name: param_value: score

    model_ranking = get_model_ranking(model_dirs)
    model_ranking.reverse()  # worst to best
    def score_ranking_based(i, param_name, param_value):
        """
        score settings depending on the ranking of the model
        eg: best of 32 models has batch_size 10 -> batch_size 10 gets 32 points
        """
        param_value = cleanup_str(param_value)
        if not param_name in settings_ranking.keys():
            settings_ranking[param_name] = {}
        if not param_value in settings_ranking[param_name].keys():
            settings_ranking[param_name][param_value] = 0
        settings_ranking[param_name][param_value] += i      # i+1 is reverse place in the ranking, worst model is at i=0

    def score_accuracy_based(i, param_name, param_value):
        """
        score settings depending on the accuracy of the model
        eg: models has batch_size 10 and accuracy 63% -> batch_size 10 gets 63 points
        """
        param_value = cleanup_str(param_value)
        if not param_name in settings_ranking.keys():
            settings_ranking[param_name] = {}
        if not param_value in settings_ranking[param_name].keys():
            settings_ranking[param_name][param_value] = 0
        settings_ranking[param_name][param_value] += int(model_ranking[i][1])  # accuracy

    if use_ranking_instead_of_accuracy:
        score = lambda i, name, val : score_ranking_based(i, name, val)
    else:
        score = lambda i, name, val : score_accuracy_based(i, name, val)
    for i in range(len(model_ranking)):
        st = mio.load_settings(model_ranking[i][0])
        score(i, "num_features", st.num_features)
        score(i, "num_layers", st.num_layers)
        score(i, "hidden_size", st.hidden_size)
        score(i, "num_epochs", st.num_epochs)
        score(i, "bidirectional", st.bidirectional)
        score(i, "optimizer", st.optimizer)
        score(i, "scheduler", st.scheduler)
        score(i, "loss_func", st.loss_func)
        score(i, "transforms", st.transforms)
        score(i, "splitter", st.splitter)
        score(i, "batch_size", st.batch_size)
    # remove parameters with only one value
    settings_ranking = { k: v for k, v in settings_ranking.items() if len(v) > 1 }
    cache["settings_ranking"] = settings_ranking.copy()
    return settings_ranking

def get_settings_ranking_md(model_dirs):
    """
    turn the scores dict from rank_settings into a markdown string
    """
    settings_ranking = get_settings_ranking(model_dirs)
    s = ""
    for param_name, d in settings_ranking.items():
        s += f"- {param_name}:\n"
        sorted_scores = sorted(d.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(sorted_scores)):
            param_value, score = sorted_scores[i]
            s += f"\t{i+1}. `{param_value}` ({score} points)\n"
    return s




def interactive_model_inspector(models_dir: str):
    model_dirs = get_model_dirs(models_dir)
    model_dirs.sort()
    model_names = [ mdir[mdir.rfind('/')+1:] for mdir in model_dirs ]

    def print_options():
        s = fill_and_center("Interactive Model Inspector") + "\n"
        for i in range(len(model_names)):
            s += f"{i+1:02}: {model_names[i]}\n"
        s += """    ---
x:  print model info for x. listed model (1-based)
x.: print model info for x. ranked model (1-based)
w:  write last model info
wa: write info for all listed models
q:  quit
*:  name of model or path to model directory. If not found, reprint list."""
        print(s)
    last_model_info = None
    last_model_dir = None

    def print_model_info(model_dir):
        last_model_dir = model_dir
        last_model_info = get_model_info_md(last_model_dir)
        print(last_model_info)
    print_options()
    loop = True
    try:
        while loop:
            answer = input("> ")
            if len(answer) == 0: continue
            try:  # if x -> take x. from listed models
                i = int(answer)
                if 0 < i and i <= len(model_dirs):

                    print_model_info(model_dirs[i-1])
                    continue
            except ValueError: pass
            if answer.endswith('.'):  # if x. -> take x. from model ranking
                try:
                    i = int(answer[:-1])
                    if 0 < i and i <= len(model_dirs):
                        model_ranking = get_model_ranking(model_dirs)
                        print_model_info(model_ranking[i-1][0])
                        continue
                except ValueError: pass

            elif answer == "w":
                if last_model_info is None:
                    print("Print a model info first.")
                    continue
                write_model_info(last_model_dir, last_model_info)
            elif answer == "wa":
                for model_dir in model_dirs:
                    write_model_info(model_dir)
            elif answer == "q":
                loop = False
                continue
            else:
                if path.isdir(answer):  # if model dir
                    print_model_info(answer)
                elif path.isdir(f"{models_dir}/{answer}"):  # if model name
                    print_model_info(f"{models_dir}/{answer}")
                else:
                    print(f"'{answer}' is not a model name in {models_dir} or path to a model directory.")
                    print_options()
    except KeyboardInterrupt:  # if <C-C>
        pass
    except EOFError:  # if <C-D>
        exit(0)
    return True



def main():
    if len(argv) != 2:
        print(f"Exactly one argument (models directory) is required, but got {len(argv)-1}.")
        exit(1)

    # models_dir = "/home/matth/Uni/TENG/models_phase_2"  # where to save models, settings and results
    models_dir = path.abspath(path.expanduser(argv[1]))

    model_dirs = get_model_dirs(models_dir)

    def save_model_ranking():
        model_ranking = get_model_ranking_md(model_dirs)
        with open(f"{models_dir}/ranking_models.md", "w") as file:
            file.write(model_ranking)

    def save_settings_ranking():
        scores = get_settings_ranking(model_dirs)
        with open(f"{models_dir}/ranking_settings.md", "w") as file:
            file.write(get_settings_ranking_md(scores))


    # if the functions return True, the options are printed again
    options = {
        '1':    ("Print model ranking", lambda: print(get_model_ranking_md(model_dirs))),
        '2':    ("Save model ranking", save_model_ranking),
        '3':    ("Print settings ranking", lambda: print(get_settings_ranking_md(model_dirs))),
        '4':    ("Save settings ranking", save_settings_ranking),
        '5':    ("Interactive model inspector", lambda: interactive_model_inspector(models_dir)),
        '6':    ("Resave all images", lambda: resave_images_svg(model_dirs)),
        'q':    ("quit", exit)
    }

    def print_options():
        print(fill_and_center("Model Evaluator"))
        for op, (name, _) in options.items():
            print(f"{op:4}: {name}")
    print(f"Using models directory '{models_dir}', which contains {len(model_dirs)} models")
    print_options()
    try:
        while True:
            answer = input("> ")
            if answer in options.keys():
                reprint = options[answer][1]()
                if reprint == True: print_options()
            else:
                print(f"Invalid option: '{answer}'")
                print_options()
    except KeyboardInterrupt: pass
    except EOFError: pass

if __name__ == "__main__":
    main()
