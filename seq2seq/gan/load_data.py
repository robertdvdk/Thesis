"""
Author: Robert van der Klis

Loads data necessary for sequence to sequence learning

Usage: python3 load_data.py
"""


# Import statements
import os

# Function definitions
def load_seqpairs(processed_dir, begin_base, end_base):
    """Loads promoter-protein sequence pairs from the processed files directory

    Args:
        processed_dir::str
            the directory containing the processed files
        begin_base::int
            the base at which to begin (1000 is Transcription Start Site)
        end_base::int
            the base at which to end (1000 is Transcription Start Site)

    Returns:
        domains_dict::dict
            dictionary containing all processed data:
                {UniProt ID: (promoter sequence: [domains])}
    """
    seqs_dict = {}
    for file in os.listdir(f'{processed_dir}/domains{begin_base}-{end_base}'):
        with open(f'{processed_dir}/paired_sequences{begin_base}-{end_base}/{file}') \
                as fopen:
            curr_upid, curr_promseq, curr_protseq = '', '', ''
            for line in fopen:
                words = line.split()
                if words[0].startswith('UPID'):
                    curr_upid = words[1]
                if words[0].startswith('PROMSEQ'):
                    curr_promseq = words[1]
                if words[0].startswith('PROTSEQ'):
                    if len(words[1]) > 200:
                        curr_protseq = words[1][:200]
                    else:
                        curr_protseq = words[1]
                if curr_upid and curr_promseq and curr_protseq:
                    seqs_dict[curr_upid] = (curr_promseq, curr_protseq)
                    curr_upid = ''
                    curr_promseq = ''
                    curr_protseq = ''
    return seqs_dict

def write_seqsdict(outfile):
    seqsdict = load_seqpairs('/home/klis004/nbk_lustre/processed_data', 800, 1000)
    with open(outfile, 'a') as fopen:
        for v in seqsdict.values():
            fopen.write(f'{v[0]}\t{v[1]}\n')

def prep_gan(infile):
    with open(infile, 'r') as fopen:
        lines = fopen.readlines()

    lines = [i.split()[0] + "\n" for i in lines]

    num_lines = len(lines)
    train_idx, valid_idx = int(0.7*num_lines), int(0.85*num_lines)
    train_data = lines[:train_idx]
    valid_data = lines[train_idx:valid_idx]
    test_data = lines[valid_idx:]
    train_cats = ['1\n' for i in train_data]
    with open('train_data.txt', 'w') as fopen:
        fopen.writelines(train_data)
    with open('valid_data.txt', 'w') as fopen:
        fopen.writelines(valid_data)
    with open('test_data.txt', 'w') as fopen:
        fopen.writelines(test_data)
    with open('train_cats.txt', 'w') as fopen:
        fopen.writelines(train_cats)


def main():
    # write_seqsdict('realdna-aa_short_all.txt')
    prep_gan('tensorflow_data_3ediff_constanteachtergrond.txt')

if __name__ == "__main__":
    main()
