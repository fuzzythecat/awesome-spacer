# -*- coding: utf-8 -*-
import os
import argparse
import tensorflow as tf

from glob import glob
from random import shuffle
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

from models.config import Config
from models.utils import build_dataset
from models.spacer import Spacer

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
   
    # Data preparation
    filenames = glob(os.path.join(FLAGS.data_path, '*.txt'))
    shuffle(filenames)

    validation_split = int(len(filenames) * FLAGS.validation_split)
    val_filenames = filenames[:validation_split]
    train_filenames =  filenames[validation_split:]
 
    print('Found {} files.'.format(len(filenames)))
    print('Using {} files for training.'.format(len(train_filenames)))

    print('Preparing training dataset.')
    X_train, y_train = build_dataset(train_filenames, config) 
    print('Preparing validation dataset.')
    X_val, y_val = build_dataset(val_filenames, config)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1024)
    train_dataset = train_dataset.batch(config.BATCH_SIZE)
    train_dataset = train_dataset.repeat()

    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    valid_dataset = valid_dataset.batch(config.BATCH_SIZE)
    valid_dataset = valid_dataset.repeat()
    
    # Model loading
    model = Spacer(FLAGS.trained_model,
                   config)

    optimizer = optimizers.Adam(
            config.LEARNING_RATE,
            False)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy',
                           tf.metrics.Precision(),
                           tf.metrics.Recall(),
                           tf.metrics.AUC()])

    model.summary()

    MODEL_SAVE_DIR = 'outputs/'
    if not os.path.exists(MODEL_SAVE_DIR):
        os.mkdir(MODEL_SAVE_DIR)

    model_path = os.path.join(MODEL_SAVE_DIR,
            '{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5')
    
    checkpoint = ModelCheckpoint(filepath=model_path, 
                                 monitor='val_loss',
                                 verbose=1,
                                 save_weights_only=True,
                                 save_best_only=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=config.EPOCHS,
                        batch_size=config.BATCH_SIZE,
                        callbacks=[checkpoint],
                        verbose=1)
