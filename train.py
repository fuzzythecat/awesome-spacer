# -*- coding: utf-8 -*-
import os
import argparse
import tensorflow as tf

from glob import glob
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

from models.config import Config
from models.utils import build_dataset
from models.utils import build_model
from models.utils import TextHistory

def str2bool(v):
    return v.lower() in ('yes', 'y', 'true', 't')


parser = argparse.ArgumentParser()
# Training configurations
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--trained_model', type=str, default=None)
#parser.add_argument('--logs', type=str, default='./logs')

# Data configurations
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--validation_split', type=float, default=.2)
parser.add_argument('--shuffle_data', type=str2bool, default=True)

FLAGS = parser.parse_args()
config = Config()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
   
    # Model loading
    strategy = tf.distribute.MirroredStrategy()
    gpu_nums = strategy.num_replicas_in_sync

    if gpu_nums > 1:
        # For multi-GPU training.
        with strategy.scope():
            model = build_model(FLAGS.trained_model, 
                                config)
    else:
        # For single GPU training.
        model = build_model(FLAGS.trained_model, 
                            config)
    model.summary()

    # Data preparation
    filenames = glob(os.path.join(FLAGS.data_path, '*.txt'))
    train_filenames, val_filenames = train_test_split(filenames, 
            test_size=FLAGS.validation_split)

    print('Found {} files.'.format(len(filenames)))
    print('Using {} files for training.'.format(len(train_filenames)))

    print('Preparing training dataset.')
    X_train, y_train = build_dataset(train_filenames, config) 
    print('Preparing validation dataset.')
    X_val, y_val = build_dataset(val_filenames, config)

    # Integrating with tf.data
    global_batch_size = config.BATCH_SIZE * gpu_nums
    train_steps = len(X_train) // global_batch_size
    valid_steps = len(X_val) // global_batch_size

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(global_batch_size).repeat()
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    valid_dataset = valid_dataset.batch(global_batch_size).repeat()  

    # Model training.
    MODEL_SAVE_DIR = 'outputs/'
    if not os.path.exists(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)

    model_path = os.path.join(MODEL_SAVE_DIR,
            '{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5')
   
    # Define callbacks.
    checkpoint = ModelCheckpoint(filepath=model_path, 
                                 monitor='val_loss',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 verbose=1)
    
    tensorboard = TensorBoard(MODEL_SAVE_DIR,
                              write_graph=True,
                              update_freq=300)
    
    text_logger = TextHistory(X_val[0:5],
                              config=config,
                              log_dir=MODEL_SAVE_DIR)

    callbacks = [checkpoint, tensorboard, text_logger]


    history = model.fit(train_dataset,
                        validation_data=valid_dataset,
                        steps_per_epoch=train_steps,
                        validation_steps=valid_steps,
                        epochs=config.EPOCHS,
                        callbacks=callbacks,
                        verbose=1)
