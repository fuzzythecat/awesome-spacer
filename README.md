# awesome-spacer

awesome-spacer is a project for automatic Korean word spacing, using **TensorFlow 2** + **Keras**.

## Requirements

- `Python 3.6`
- `TensorFlow 2.0`
- `NumPy`
- `tqdm`
- `scikit-learn`

## Getting Started
* [awesome-spacer-train-colab.ipynb](notebooks/awesome-spacer-train-colab.ipynb): You can use this to train your own model on Sejong corpus with Google Colab. To train on custom datasets, try using CLI instead.  

* [awesome-spacer-test-colab.ipynb](notebooks/awesome-spacer-test-colab.ipynb): You can use this to test pre-trained models trained on Sejong corpus. Weight links and corresponding model configurations are included in the notebook.

## Train

To train the model, you should provide the path to your dataset.
You can import this module and train in Jupyter Notebook(see notebooks for example), 
or train from CLI. 

Currently training with CPU is **not** supported.
```
# Train a new model from scratch. 
python train.py --data_path path/to/dataset --gpu_list 0

# Continue training from pre-trained model.
python train.py --data_path path/to/dataset --gpu_list 0 --trained_model path/to/weights.h5

# Pass GPU ids for multi-GPU training.
python train.py --data_path path/to/dataset --gpu_list 0,1,2,3
```

## Workshop Materials
* Hands-on Workshop @ [2019 Global AI Bootcamp - Seoul](https://festa.io/events/772)
