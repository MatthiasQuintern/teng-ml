# Machine learning for material recognition with a triboelectric nanogenerator (TENG)
This project was written for my bachelor's thesis.

It was written to classify TENG voltage output from pressing it against different materials.
Contents:
- Data preparation/plotting/loading utilites
- (Bi)LSTM + fully connected + softmax model for name classifiying TENG output
- Progress tracking utilities to easily find the best parameters

## Model training
Adjust the parameters in `main.py` and run it.
All models and the settings they were trained with are automatically serialized with pickle and stored in a subfolder
of the `<model_dir>` that was set in `main.py`.


## Model evaluation
Run `find_best_model.py <model_dir>` with the `<model_dir>` specified in `main.py` during training.
