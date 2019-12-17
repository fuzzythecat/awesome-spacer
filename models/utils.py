import tensorflow as tf

from tqdm import tqdm
from tensorflow import metrics
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models.spacer import Spacer


def encode_string(s, chr_to_idx):
    encoded = []
    for c in s:
        try:
            idx = chr_to_idx[c]
        except:
            idx = chr_to_idx['<UNK>']
        encoded.append(idx)
    return encoded
    
    
def get_label(s, idx=0):
    label = []
    while True:
        try:
            next_ch = s[idx + 1]
        except:
            # End of sentence
            label.append(0)
            break
        if next_ch == ' ':
            label.append(1)
            idx += 2
        else:
            label.append(0)
            idx += 1
    return label
    
    
def build_dataset(filenames, config=None):
    X = []
    y = []
    
    for p in tqdm(filenames):
        with open(p, 'r', encoding='utf-8') as f:
            texts = f.readlines()
            # Remove redundant whitespaces(noise),
            # convert to lowercase,
            # and filter out zero length input sentences.
            texts = [' '.join(t.split()).lower() for t in texts if t]

            # Compute labels.
            labels = [get_label(t) for t in texts]
    
            # Remove whitespaces.
            texts = [''.join(t.split()) for t in texts]
            
            # Encode texts.
            texts = [encode_string(t, config.CHR_TO_IDX) for t in texts]
            
            X += texts
            y += labels
    
    X = pad_sequences(X, maxlen=config.MAX_TEXT_LEN, 
                      padding='post', truncating='post',
                      value=config.CHR_TO_IDX['<PAD>'])
    y = pad_sequences(y, maxlen=config.MAX_TEXT_LEN, 
                      padding='post', truncating='post',
                      value=0)
    
    return (X, y)


def build_model(weights=None, config=None):
    '''Returns a compiled model.'''
    model = Spacer(weights, config)

    lr_schedule = optimizers.schedules.ExponentialDecay(
            config.LEARNING_RATE,
            decay_steps=config.DECAY_STEPS,
            decay_rate=config.DECAY_RATE,
            staircase=True)
    optimizer = optimizers.Adam(lr_schedule, False)

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy',
                           metrics.Precision(),
                           metrics.Recall(),
                           metrics.AUC()])

    return model


class TextHistory(Callback):
    def __init__(self, data=None, update_freq=300,
                 log_dir='logs', config=None):
        self.data = data
        self.config = config
        self.update_freq = update_freq
        self.writer = tf.summary.create_file_writer(log_dir)
        self.last_step = 0

    def on_batch_end(self, batch, logs=None):
        if batch % self.update_freq != 0:
            return
        # Run prediction.
        probs = self.model.predict(self.data)

        # Only log test results every update_freq steps.
        texts = []
        for i, X in enumerate(self.data):
            text = ''
            for j, idx in enumerate(X):
                if idx != self.config.CHR_TO_IDX['<PAD>']:
                    # Skip paddings.
                    text += self.config.IDX_TO_CHR[idx]
                if probs[i][j] >= 0.5:
                    text += ' '
            texts.append(text)
        self.last_step += 1

        with self.writer.as_default():
            for i, t in enumerate(texts):
                tf.summary.text('text sample #{}'.format(i), t, 
                       self.last_step * self.update_freq)
