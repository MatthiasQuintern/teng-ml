from ..tracker.epoch_tracker import EpochTracker
from ..util.settings import MLSettings
import pickle


"""
Load and save model, settings and EpochTrackers from/on disk
"""

def load_tracker_validation(model_dir):
    with open(f"{model_dir}/tracker_validation.pkl", "rb") as file:
        validation_tracker: EpochTracker = pickle.load(file)
    return validation_tracker

def load_tracker_training(model_dir):
    with open(f"{model_dir}/tracker_training.pkl", "rb") as file:
        training_tracker: EpochTracker = pickle.load(file)
    return training_tracker

def load_settings(model_dir):
    with open(f"{model_dir}/settings.pkl", "rb") as file:
        st: MLSettings = pickle.load(file)
    return st

def load_model(model_dir):
    with open(f"{model_dir}/model.pkl", "rb") as file:
        model = pickle.load(file)
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
