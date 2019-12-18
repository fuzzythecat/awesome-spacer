# -*- coding: utf-8 -*-

import os
import argparse
from glob import glob
from tqdm import tqdm


def concat_string(s, maxlen):
    # Credits: EthanJYK (https://github.com/EthanJYK)
    i = 1
    while i < len(s):
        if len(s[i-1]) + len(s[i]) < maxlen:
            s[i-1] += f' {s.pop(i)}'
        else:
            i += 1


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--max_text_len', type=int, default=200)

FLAGS = parser.parse_args()


if __name__ == '__main__':
    filepaths = glob(os.path.join(FLAGS.data_path, '*.txt'))
    print('Processing {} files.'.format(len(filepaths)))

    len_raw = 0
    len_processed = 0

    for path in tqdm(filepaths):
        with open(path, 'r') as f:
            texts = f.readlines()

        # Remove trailing whitespaces.
        texts = [text.strip() for text in texts]
        len_raw += len(texts)
            
        # Concat strings.
        concat_string(texts, FLAGS.max_text_len)
        len_processed += len(texts)

        # Overwrite file contents.
        with open(path, 'w') as f:
            for text in texts:
                f.write('%s\n' % text)

    print('Line length: {} -> {}'.format(len_raw, len_processed))
