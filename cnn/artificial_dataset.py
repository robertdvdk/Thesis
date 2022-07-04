"""
Generates artificial promoter-protein pairs with motifs that have a
specified level of difficulty

Author: Robert van der Klis
"""
# Import statements
from random import choice
from random import random
from random import seed

# Global variables
NUCLEOTIDES = ["A", "C", "G", "T"]
AMINO_ACIDS = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N",
               "P", "Q", "R", "S", "T", "V", "W", "Y"]
DOMAINS = ['PFAM1', 'PFAM2', 'PFAM3', 'PFAM4', 'PFAM5', 'PFAM6', 'PFAM7',
           'PFAM8', 'PFAM9', 'PFAM10']

# Function definitions
def similarity(string1, string2):
    """Calculates the proportion of identical characters in two strings

    Args:
        string1::str
            First string to use in the comparison
        string2::str
            Second string to use in the comparison

    Returns:
        identity::float
            Proportion of characters in the strings that are identical
    """
    identical_chars = 0
    length_shortest = min(len(string1), len(string2))
    for char in range(length_shortest):
        if string1[char] == string2[char]:
            identical_chars += 1
    identity = identical_chars / length_shortest
    return identity

def generate_motif_pairs(promoter_motif_length, aa_motif_length, n_motifs):
    """Generates artificial promoter-protein motif pairs

    Args:
        promoter_motif_length::int
            Length of the promoter motifs to generate
        aa_motif_length::int
            Length of the amino acid motifs to generate
        n_motifs::int
            Number of motif pairs to generate

    Returns:
        motif_pairs::dict
            A dictionary of {promoter motif : protein motif}
    """
    motif_pairs = {}
    i = 0
    while i < n_motifs:
        nucleotide_motif = ""
        for j in range(promoter_motif_length):
            nucleotide_motif += choice(NUCLEOTIDES)
        aa_motif = ""
        for j in range(aa_motif_length):
            aa_motif += choice(AMINO_ACIDS)
        for promoter, protein in motif_pairs.items():
            if similarity(nucleotide_motif, promoter) > 0.75:
                continue
            elif similarity(aa_motif, protein) > 0.75:
                continue
        motif_pairs[nucleotide_motif] = aa_motif
        i += 1
    return motif_pairs

def generate_motif_dataset(motif_chance, motif_pairs, num_sequences, prom_length,
                     prot_length):
    """Generates an artificial dataset of promoter-protein pairs

    Args:
        motif_chance::float
            Chance to insert a motif at each place in the promoter sequence
        motif_pairs::dict
            A dictionary of {promoter motif : protein motif}
        num_sequences::int
            The number of promoter-protein pairs to generate (excl motifs)
        prom_length::float
            The length of the promoter sequence to generate (excl motifs)
        prot_length::float
            The length of the protein sequence to generate

    Returns:
        promoter_protein::dict
            A dictionary of {promoter sequence : protein sequence},
            artificially generated sequences with random motifs in them
    """
    promoter_protein = {}
    for pair in range(num_sequences):
        promoter, protein = "", "M"
        used_motifs = []
        for base in range(prom_length):
            if random() < motif_chance:
                chosen_motif = choice(list(motif_pairs.keys()))
                promoter += chosen_motif
                used_motifs.append(motif_pairs[chosen_motif])
            promoter += choice(NUCLEOTIDES)
        for aa in range(prot_length - 1):
            if random() < motif_chance/(prom_length/prot_length) and used_motifs:
                protein += choice(used_motifs)
            protein += choice(AMINO_ACIDS)
        promoter_protein[promoter] = protein
    return promoter_protein

def domain_pairs(promoter_motif_length, n_motifs):
    pairs = {}
    i = 0
    while i < n_motifs:
        nucleotide_motif = ""
        for j in range(promoter_motif_length):
            nucleotide_motif += choice(NUCLEOTIDES)
        new_domain = choice(DOMAINS)
        for promoter, domain in pairs.items():
            if similarity(nucleotide_motif, promoter) > 0.75:
                break
            elif new_domain == domain:
                break
        else:
            pairs[nucleotide_motif] = new_domain
            i += 1
    return pairs

def generate_domain_dataset(motif_chance, domain_pairs, num_sequences, prom_length, motif_length):
    promoter_domain = {}
    for pair in range(num_sequences):
        if pair % 10000 == 0:
            print(f'{pair} sequences generated')
        promoter = ''
        domains = set()
        while len(promoter) < prom_length:
            if random() < motif_chance and len(promoter) + motif_length < prom_length:
                chosen_motif = choice(list(domain_pairs.keys()))
                promoter += chosen_motif
                domains.add(domain_pairs[chosen_motif])
            else:
                promoter += choice(NUCLEOTIDES)
        promoter_domain[promoter] = (promoter, list(domains))
    return promoter_domain


if __name__ == "__main__":
    """The main function of this module"""
    seed(1)
    # motif_pairs = generate_motif_pairs(5, 10, 10)
    # promoter_protein_pairs = generate_motif_dataset(0.1, motif_pairs, 10000, 100, 200)
    pairs = domain_pairs(5, 10)
    print(generate_domain_dataset(0.1, pairs, 10, 50, 5))