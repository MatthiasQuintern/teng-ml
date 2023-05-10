# Machine learning for material recognition with a TENG
(Bi)LSTM for name classification.  
More information on the project are [on my website](https://quintern.xyz/en/teng.html).

## Model training
Adjust the parameters in `main.py` and run it.  
All models and the settings they were trained with are automatically serialized with pickle and stored in a subfolder
of the `<model_dir>` that was set in `main.py`.


## Model evaluation
Run `find_best_model.py <model_dir>` with the `<model_dir>` specified in `main.py` during training.

