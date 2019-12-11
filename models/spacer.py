from tensorflow.keras import layers
from tensorflow.keras import models


def _obtain_input_shape(max_text_len):
    '''Internal utility to validate the model's input shape
    
    # Arguments
        max_text_len: `int`. Maximum number of characters per sentence.

    # Returns
        An integer shape tuple.

    # Raises
        ValueError: In case of invalid argument values.
    '''
    if not (isinstance(max_text_len, int)):
        raise(ValueError('`max_text_len` should be' +
                         'a single integer.'))
    input_shape = (max_text_len, )
        
    return input_shape


def _conv_block(inputs,
                filter_nums,
                filter_sizes):
    '''Adds a convolution block.

    A convolution block consists of a group of
    1D Conv layers, applied on the same input.

    # Arguments
        inputs: Input tensor of shape `(MAX_TEXT_LEN, EMBEDDING_DIM)`.
        filter_nums: List of `int`. Determines the number of N-grams features 
            that are learned. Each element in the list must have a
            corresponding `N` parameter in `filter_sizes`.
        filter_sizes: List of `int`. Determines N in N-grams.
            Each element in the list must have a corresponding 
            filter configuration in `filter_nums`.

    # Raises
        ValueError: In case of invalid
            filter configurations.

    # Returns
        A list of Conv1D layers. 
    '''
    if not all(isinstance(x, int) for x in filter_nums):
        raise(ValueError('`filter_nums` should be a list ' +
                         'of integers.'))
    if not all(isinstance(x, int) for x in filter_sizes):
        raise(ValueError('`filter_sizes` should be a list ' +
                         'of integers.'))
    if not len(filter_nums) == len(filter_sizes):
        raise(ValueError('`filter_nums` and `filter_sizes` should ' +
                         'have the same size.'))

    conv_blocks = []    
    for fn, fs in zip(filter_nums, filter_sizes):
        conv = layers.Conv1D(filters=fn,
                             kernel_size=fs,
                             padding='same',
                             activation='relu',
                             strides=1)(inputs)

        conv_blocks.append(conv)

    return conv_blocks


def Spacer(weights=None,
           config=None):
    '''
    # Arguments
        max_text_len: `int`. Maximum number of characters per sentence.
        num_words: `int`. Maximum number of words to keep. 

    # Returns
        A Keras model instance.

    # Raises
        ValueError: In case of invalid
            filter configurations.

    '''
    input_shape = _obtain_input_shape(config.MAX_TEXT_LEN)

    inputs = layers.Input(shape=input_shape)
    x = layers.Embedding(config.NUM_WORDS, 
                         config.EMBEDDING_DIM,
                         input_length=config.MAX_TEXT_LEN)(inputs)

    conv_blocks = _conv_block(x, config.FILTER_NUMS,
                                 config.FILTER_SIZES)

    x = layers.Concatenate()(conv_blocks)
    x = layers.Bidirectional(layers.LSTM(100, dropout=0.3, return_sequences=True))(x)
    x = layers.LSTM(50, dropout=0.1, return_sequences=True)(x)

    x = layers.TimeDistributed(layers.Dense(300, activation='relu'))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.TimeDistributed(layers.Dense(150, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(x)
    x = layers.Reshape(input_shape)(x)

    model = models.Model(inputs, x)
 
    return model
