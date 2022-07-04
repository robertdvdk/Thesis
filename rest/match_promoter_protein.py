"""
Matches FASTA files of promoters pulled from
https://epd.epfl.ch/EPDnew_select.php with GenBank files of proteins pulled
from https://ftp.ncbi.nih.gov/genomes/refseq/

Author: Robert van der Klis
"""

# Import statements
import re
import os

# Function definitions
def parse_epd(filename):
    """Parses EPDnew fasta files and adds them to a dictionary

     Args:
        filename::str
            Name of the EPDnew fasta file

    Returns:
        seq_dict::dict
            A dictionary of {gene id: [promoter sequence, ]}, where the
            empty spot is reserved for the protein sequence
    """
    epd_dict = {}
    with open(filename, 'r') as fopen:
        for line in fopen:
            if line.startswith('>'):
                curr_id = line.split()[1][:-2]
                epd_dict[curr_id] = [""]
            else:
                epd_dict[curr_id][0] += line.strip()
    return epd_dict

def parse_genbank(filename, plant):
    """Parses UniProt fasta files and adds protein sequences to seq_dict

    Args:
        filename::str
            Name of the UniProt fasta file
        plant::bool
            Whether or not the current file is a plant; different
            identifiers are used for plants... for some reason

    Returns:
        seq_dict::dict
            Dictionary of {gene id: protein sequence}
    """
    genbank_dict = {}
    if plant:
        identifier_line = '/locus_tag='
    else:
        identifier_line = '/gene='
    with open(filename, 'r') as fopen:
        sequence_started, aa_seq = 0, ""
        for line in fopen:
            curr = line.strip()
            if curr.startswith(identifier_line):
                # line with gene identifier starts with /gene=
                m = re.search(r'".*"', line)
                gene_id = m.group(0)[1:-1]
            elif curr.startswith('ORIGIN'):
                # "ORIGIN" always occurs directly before protein sequence in
                # GenBank files
                sequence_started = 1
            elif curr == "//":
                sequence_started = 0
                if gene_id in genbank_dict:
                    # Only keep the first protein sequence for each gene id
                    aa_seq = ""
                    continue
                else:
                    genbank_dict[gene_id] = aa_seq
                    aa_seq = ""
            elif sequence_started == 1:
                aa_seq += "".join(curr.split()[1:])
                # AA seqs are separated by spaces -> need to join
    return genbank_dict

def pair_epd_genbank(epd_dict, genbank_dict):
    """Pairs EPD and GenBank sequences, and removes entries with no protein

    Args:
        epd_dict::dict
            Dictionary of {gene id: [promoter sequence, ]}
        genbank_dict::dict
            Dictionary of {gene id: protein sequence}

    Returns:
        epd_dict::dict
            Dictionary of {gene id: [promoter sequence, protein sequence]}
    """
    to_remove = []
    for protein_id in epd_dict.keys():
        if protein_id in genbank_dict:
            epd_dict[protein_id].append(genbank_dict[protein_id])
        else:
            to_remove.append(protein_id)
    for id in to_remove:
        del epd_dict[id]
    return epd_dict

def generate_prom_prot_dict(filename_epd, filename_genbank, species):
    """Generates dict of protein ID, with corresponding promoters and proteins

    Args:
        filename_epd::str
            Name of the EPDnew promoter fasta file
        filename_uniprot::str
            Name of the UniProt protein fasta file

    Returns:
        seq_dict::dict
            Dictionary of {protein id: [promoter sequence, protein sequence]},
            where entries that do not have both promoter and protein sequences
            have been removed
    """
    if species == 'araTha1':
        plant = True
    else:
        plant = False
    epd_dict = parse_epd(filename_epd)
    genbank_dict = parse_genbank(filename_genbank, plant)
    seq_dict = pair_epd_genbank(epd_dict, genbank_dict)
    return seq_dict

def make_complete_dict(epd_dir, genbank_dir, out_dir):
    seq_dict = {}
    # Iterate over all species
    for kingdom in os.listdir(epd_dir):
        for species_filename in os.listdir(epd_dir + "/" + kingdom):
            species = species_filename.split(".")[0]
            if species == 'pfa2' or species == 'zm3' or species == 'ce6':
                # These two have different genbank file formats, skip for now
                continue
            if os.path.exists(f'{out_dir}/{kingdom}/{species}.txt'):
                continue
            print(kingdom, species)
            epd_file = f"{epd_dir}/{kingdom}/{species}.fa"
            genbank_file = f"{genbank_dir}/{kingdom}/{species}.gpff"
            seq_dict[species] = generate_prom_prot_dict(epd_file, genbank_file, species)
    return seq_dict

def main(epd_dir, genbank_dir, out_dir):
    seq_dict = make_complete_dict(epd_dir, genbank_dir, out_dir)
    for kingdom in os.listdir(epd_dir):
        for species_filename in os.listdir(epd_dir + "/" + kingdom):
            species = species_filename.split(".")[0]
            # Check if directory has to be made
            if not os.path.exists(f'{out_dir}'):
                os.mkdir(f'{out_dir}')
                os.mkdir(f'{out_dir}/{kingdom}')
            elif not os.path.exists(f'{out_dir}/{kingdom}'):
                os.mkdir(f'{out_dir}/{kingdom}')
            elif os.path.exists(f'{out_dir}/{kingdom}/{species}.txt'):
                # Do nothing if output file already exists
                continue
            if species == 'pfa2' or species == 'zm3' or species == 'ce6':
                # These two have different genbank file formats
                continue
            with open(f'{out_dir}/{kingdom}/{species}.txt', 'w') as fopen:
                for gene_id, (prom_seq, prot_seq) in seq_dict[species].items():
                    fopen.write(f">{gene_id}\n{prom_seq}\n{prot_seq}\n")

if __name__ == "__main__":
    """The main function of this module"""
    main("/home/klis004/nbk_lustre/raw_data/epd",
         "/home/klis004/nbk_lustre/raw_data/genbank/eukaryotes",
         "/home/klis004/nbk_lustre/processed_data/eukaryotes")