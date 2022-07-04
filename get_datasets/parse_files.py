"""
Contains parsers for EPDnew fasta files and GenBank files, used as helper
module

Author: Robert van der Klis
"""
# Import statements
import os
import sys

# Function definitions
def parse_epd(filename, begin_base, end_base):
    """Parses EPDnew fasta files and adds them to a dictionary

     Args:
        filename::str
            Name of the EPDnew fasta file
        begin_base::int
            The index of the first promoter base to include. Min: 0, max: 1251
        end_base::int
            The index of the last promoter base to include. Min: 0, max: 1251

    Returns:
        seq_dict::dict
            A dictionary of {gene id: [promoter sequence]}
    """
    epd_dict = {}
    with open(filename, 'r') as fopen:
        curr_seq = ""
        for line in fopen:
            if line.startswith('>'):
                full_id = line.split()[1]
                # Remove underscore and unnecessary number at the end
                if full_id.endswith('_'):
                    curr_id = line.split()[1][:-1]
                else:
                    curr_id = line.split()[1][:-2]
                if curr_seq:
                    epd_dict[curr_id] = curr_seq[begin_base:end_base]
                    curr_seq = ""
            else:
                curr_seq += line.strip()
    return epd_dict

def parse_genbank(filename):
    """Parses UniProt fasta files and adds protein sequences to seq_dict

    Args:
        filename::str
            Name of the UniProt fasta file

    Returns:
        seq_dict::dict
            Dictionary of {gene id: protein sequence}
    """
    genbank_dict = {}
    with open(filename, 'r') as fopen:
        sequence_started, aa_seq = 0, ""
        for line in fopen:
            curr = line.strip()
            if curr.startswith('VERSION'):
                # line with gene identifier starts with /gene=
                id = curr.split()[1]
            elif curr.startswith('ORIGIN'):
                # "ORIGIN" always occurs directly before protein sequence in
                # GenBank files
                sequence_started = 1
            elif curr == "//":
                sequence_started = 0
                genbank_dict[id] = aa_seq
                aa_seq = ""
            elif sequence_started == 1:
                aa_seq += "".join(curr.split()[1:])
                # AA seqs are separated by spaces -> need to join
    return genbank_dict

if __name__ == "__main__":
    """The main function of this module"""
    pass