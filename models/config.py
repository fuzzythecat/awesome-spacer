class Config(object):
    '''Project configuration class.
    '''
    # Model configurations.
    # Maximum number of characters per sentence.
    MAX_TEXT_LEN = 200

    # Dimension of the dense word embedding.
    EMBEDDING_DIM = 64

    # Determines N in N-grams.
    # Each element in the list must have a corresponding 
    # filter configuration in `filter_nums`.
    FILTER_SIZES = [4, 6, 8]
    # Determines the number of N-grams features that 
    # are learned. Each element in the list must have a
    # corresponding `N` parameter in `filter_sizes`.
    FILTER_NUMS = [64, 64, 64] 

    # Training configurations.
    # Effective batch size per GPU.
    BATCH_SIZE = 256
    EPOCHS = 30
    
    # Learning rate and hyperparameters for
    # learning rate schedule.
    # Set learning rate to lower values
    # for transfer learning.
    # Adam is used as default.
    LEARNING_RATE = 0.001
    DECAY_STEPS = 100000
    DECAY_RATE = 0.96

    # Unicode ranges for characters in general.
    COMPLETE_KOR_RANGE = range(0xac00, 0xd7a4)
    ALPH_UPPER_RANGE = range(0x0041, 0x005b)
    ALPH_LOWER_RANGE = range(0x0061, 0x007b)
    
    # Numbers, punctuations, etc.
    NUMBERS = '0123456789'
    PUNCTUATIONS = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    # Default tokens used. 
    # '<PAD>': Added before or after each line.
    # '<START>': Added before start of every line.
    # '<UNK>': Replaces OOV characters. 
    # To add custom tokens, custom `get_label` function
    # must also be provided in `utily.py`. 
    TOKENS = ['<PAD>',
              '<START>',
              '<UNK>']
    
    # Characters not listed below
    # will be replaced with <UNK> tokens.
    VALID_CHARACTERS = [
        COMPLETE_KOR_RANGE,
        ALPH_UPPER_RANGE,
        ALPH_LOWER_RANGE,
        NUMBERS,
        PUNCTUATIONS]
    
    
    def __init__(self):
        '''Create valid character sets and mapping tables'''
        self.IDX_TO_CHR = []
        self.CHR_TO_IDX = {}

        # The first few indices are reserved.
        self.IDX_TO_CHR.extend(self.TOKENS)

        # Create idx->chr mapping.
        for r in self.VALID_CHARACTERS:
            for c in r:
                if isinstance(c, int):
                    self.IDX_TO_CHR.append(chr(c))
                elif isinstance(c, str):
                    self.IDX_TO_CHR.append(c)

        # Create chr->idx mapping.
        for v, k in enumerate(self.IDX_TO_CHR):
            self.CHR_TO_IDX[k] = v
            
        # Maximum number of words to keep.
        # i.e. maximum integer index + 1.
        self.NUM_WORDS = len(self.IDX_TO_CHR)
