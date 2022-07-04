"""
Plot histogram and output descriptive statistics of protein sequence lengths

Author: Robert van der Klis
"""
# Import statements
import os
import matplotlib.pyplot as plt
import pandas as pd

# Function definitions
def count_sequences(processed_dir):
    """Counts the number of promoter-protein pairs in each processed file

    Args:
        processed_dir::string
            The directory in which all subdirectories with processed files
            are contained

    Returns:
        prot_length::dict
            Dictionary containing the lengths of proteins in all species that
            are paired with a promoter:
                {species: list of protein lengths in species}}
    """
    domains_amt = {}
    seq_dir = f'{processed_dir}/domains'
    for file in os.listdir(seq_dir):
        species = file.split(".")[0]
        domains_amt[species] = []
        with open(f"{seq_dir}/{file}") as fopen:
            for line in fopen:
                if line.startswith('DOMAINS'):
                    domains = line.split()[1:]
                    domains_amt[species].append(len(domains))
    return domains_amt

def visualize(prot_length):
    """Visualizes protein sequence lengths of all species in a histogram,
    and outputs descriptive statistics for the lengths

    Args:
        prot_length::dict
            Dictionary containing the lengths of proteins in all species that
            are paired with a promoter: {kingdom: {species: list of protein
            lengths in species}}

    Returns:
        None
    """
    fig, ax = plt.subplots()
    list_prot_lengths = []
    for species, prot_lengths in prot_length.items():
        list_prot_lengths.extend(prot_lengths)
    ax.hist(list_prot_lengths, bins=30, stacked=True, range=(0, 30))
    plt.suptitle('Domain count distribution')
    plt.xlabel('Domain count')
    plt.ylabel('Count')
    for species, prot_lengths in prot_length.items():
        print(species)
        print(pd.Series(prot_lengths).describe())
    plt.savefig('/home/klis004/nbk_lustre/processed_data/test.png')

def main():
    """The main function of this module"""
    prot_length = count_sequences("/home/klis004/nbk_lustre/processed_data/")
    visualize(prot_length)

if __name__ == "__main__":
    main()
