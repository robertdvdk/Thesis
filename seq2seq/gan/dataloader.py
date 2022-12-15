"""
Author: Robert van der Klis

Helper module, tokenizes strings.
"""


# Import statements
import numpy as np
import collections

# Function definitions
def tokenize_string(sample):
    return tuple(sample.lower().split(' '))

def load_dataset(max_length, max_n_examples, strip=False, tokenize=False, max_vocab_size=26, data_dir=''):
    '''Adapted from https://github.com/igul222/improved_wgan_training/blob/master/language_helpers.py'''
    print ("loading dataset...")

    lines = []

    with open(data_dir, 'r') as f:
        for line in f:
            dna = line.split('\t')[0]
            if strip:
                dna = dna.strip()

            dna = tuple(dna.lower())

            lines.append(dna)
            if len(lines) == max_n_examples:
                break
    np.random.shuffle(lines)

    counts = collections.Counter(char for line in lines for char in line)

    charmap = {}
    inv_charmap = []

    for char, count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    print('dataset loaded')
    return lines, charmap, inv_charmap

def decode_one_seq(img, letter_dict = {'A':0, 'C':1, 'G':2, 'T':3}):
    seq = ''
    for row in range(len(img)):
        on = np.argmax(img[row,:])
        seq += letter_dict[on]
    return seq


def main():
    load_dataset(250, 2, data_dir='/home/klis004/nbk_lustre/gan/tensorflow/logs/3ediff_constanteachtergrond/2022.09.16-14h17m25s_lecun/samples/samples_15000')

if __name__ == "__main__":
    # Helper module, no main function
    main()
