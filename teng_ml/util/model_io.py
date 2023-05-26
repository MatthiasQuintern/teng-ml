from ..tracker.epoch_tracker import EpochTracker
from ..util.settings import MLSettings

import io
import pickle

# from https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "teng-ml" in module:
            module = module.replace("teng-ml", "teng_ml")
        return super(RenameUnpickler, self).find_class(module, name)
def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()
def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)


"""
Load and save model, settings and EpochTrackers from/on disk
"""

def load_tracker_validation(model_dir):
    with open(f"{model_dir}/tracker_validation.pkl", "rb") as file:
        validation_tracker: EpochTracker = renamed_load(file)
    return validation_tracker

def load_tracker_training(model_dir):
    with open(f"{model_dir}/tracker_training.pkl", "rb") as file:
        training_tracker: EpochTracker = renamed_load(file)
    return training_tracker

def load_settings(model_dir):
    with open(f"{model_dir}/settings.pkl", "rb") as file:
        st: MLSettings = renamed_load(file)
    return st

def load_model(model_dir):
    with open(f"{model_dir}/model.pkl", "rb") as file:
        model = renamed_load(file)
    return model


def save_tracker_validation(model_dir, validation_tracker: EpochTracker):
    with open(f"{model_dir}/tracker_validation.pkl", "wb") as file:
        pickle.dump(validation_tracker, file)

def save_tracker_training(model_dir, training_tracker: EpochTracker):
    with open(f"{model_dir}/tracker_training.pkl", "wb") as file:
        pickle.dump(training_tracker, file)

def save_settings(model_dir, st):
    with open(f"{model_dir}/settings.pkl", "wb") as file:
        pickle.dump(st, file)

def save_model(model_dir, model):
    with open(f"{model_dir}/model.pkl", "wb") as file:
        pickle.dump(model, file)
