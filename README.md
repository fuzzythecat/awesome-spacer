# awesome-spacer

awesome-spacer is a project for automatic Korean word spacing, using **TensorFlow 2** + **Keras**.

## Requirements

- `Python 3.6`
- `TensorFlow 2.0`
- `NumPy`
- `tqdm`

## Train

To train the model, you should provide path to the dataset.
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
