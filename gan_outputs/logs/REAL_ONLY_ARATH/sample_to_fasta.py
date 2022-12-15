"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
import os

# Function definitions
def convert(file):
    res = ''
    with open(f'./2022.10.12-18h15m41s_lecun/samples/{file}', 'r') as fopen:
        for idx, line in enumerate(fopen.readlines()):
            res += f'>SEQUENCE_{idx}\n{line}'
    with open(f'{file}_fasta.txt', 'w') as fopen:
        fopen.write(res)

def count(file):
    print(file)
    dct = {'a': 0, 'c': 0, 'g': 0, 't': 0}
    with open(f'./2022.10.12-18h15m41s_lecun/samples/{file}', 'r') as fopen:
        for line_1 in fopen:
            line = line_1.lower()
            dct['a'] += line.count('a')
            dct['c'] += line.count('c')
            dct['g'] += line.count('g')
            dct['t'] += line.count('t')
    print([i/(sum(dct.values())) for i in dct.values()])

def main():
    for file in os.listdir('./2022.10.12-18h15m41s_lecun/samples/'):
        if file == '.DS_Store':
            continue
        # convert(file)
        count(file)

if __name__ == "__main__":
    main()
