"""
Generates artificial promoter-protein pairs with motifs that have a
specified level of difficulty

Author: Robert van der Klis

Usage: helper module
"""
# Import statements
from random import choice, randint, random, seed

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

def motif_pairs(promoter_motif_length, aa_motif_length, n_motifs):
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
                     prot_length, motif_length, proteinmotif_length):
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
        promoter, protein = "", ""
        for nt in range(prom_length):
            promoter += 'A'
        for aa in range(prot_length - 1):
            protein += choice(AMINO_ACIDS)
        promtaken, prottaken = [], []
        for motif, proteinmotif in motif_pairs.items():
            # promoter = motif * 10
            # protein = proteinmotif * 10
            if random() < motif_chance:
                # prombegin = 50
                prombegin = randint(0, prom_length - motif_length)
                promend = prombegin + motif_length
                while (prombegin in promtaken) or (promend in promtaken):
                    prombegin = randint(0, prom_length - motif_length)
                    promend = prombegin + motif_length
                promtaken.append(list(range(prombegin, promend + 1)))

                # protbegin = 30
                protbegin = randint(1, prot_length - proteinmotif_length)
                protend = protbegin + proteinmotif_length
                while (protbegin in prottaken) or (protend in prottaken):
                    protbegin = randint(0, prot_length - proteinmotif_length)
                    protend = protbegin + proteinmotif_length
                promtaken.append(list(range(prombegin, promend + 1)))

                promoter = promoter[:prombegin] + motif + promoter[promend:]
                protein = protein[:protbegin] + proteinmotif + protein[protend:]
        promoter_protein[promoter] = (promoter, protein)
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

def generate_domain_dataset(motif_chance, domain_pairs, num_sequences,
                            prom_length, motif_length):
    promoter_domain = {}
    for pair in range(num_sequences):
        promoter = ''
        domains = set()
        for nt in range(prom_length):
            promoter += choice(NUCLEOTIDES)
        taken = []
        for motif, domain in domain_pairs.items():
            if random() < motif_chance:
                begin = randint(0, prom_length - motif_length)
                end = begin + motif_length
                while (begin in taken) or (end in taken):
                    begin = randint(0, prom_length - motif_length)
                    end = begin + motif_length
                taken.append(list(range(begin, end + 1)))
                promoter = promoter[:begin] + motif + promoter[end:]
                domains.add(domain)
            promoter_domain[promoter] = (promoter, list(domains))
    return promoter_domain


if __name__ == "__main__":
    """Helper module: no main function"""
    motif_length = 10
    num_motifs = 1
    motif_chance = 1
    prom_length = 100
    num_promoters = 10000
    domain = False
    aa_motif_length = 5
    prot_length = 50
    seed(1)
    pairs = motif_pairs(10, 5, 1)
    print(pairs)
    ds = generate_motif_dataset(motif_chance, pairs, num_promoters, prom_length, prot_length, motif_length, aa_motif_length)
    with open('/home/klis004/nbk_lustre/gan_input/tensorflow_data_3ediff_constanteachtergrond.txt', 'w') as fopen:
        lines = 0
        while lines < num_promoters:
            for v in ds.values():
                fopen.write(f'{v[0]}\n')
                lines += 1


        # while len(fopen.readlines()) < num_promoters:
        #     for v in ds.values():
        #         fopen.write(f'{v[0]}\n')
        # # for i in range(num_promoters): #weghalen
        # for v in ds.values():
        #     # fopen.write(f'{v[0]}\t{v[1]}\n')
        #     fopen.write(f'{v[0]}\n')
