"""
Loads real and artificial data into datasets PyTorch can work with

Author: Robert van der Klis

Usage: helper module
"""

# Import statements
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from artificial_dataset import domain_pairs, generate_domain_dataset

# Function definitions
class DomainDataset(Dataset):
    """Class designed to work with datasets as generated in
    get_datasets/main.py
    """
    def __init__(self, domains_dict, prom_length, device, domain_token_dict,
                 num_domains):
        super().__init__()
        self.domains_dict = domains_dict
        self.domain_token_dict = domain_token_dict
        self.prom_length = prom_length
        self.num_domains = num_domains
        seqs, labs = self.tokenize_dataset()
        seqs = F.one_hot(seqs.to(torch.int64))
        seqs, labs = seqs.to(device), labs.to(device)
        self.data = [(x.float(), y) for x, y in zip(seqs, labs)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def tokenize_dna(self, seq):
        """Converts DNA nucleotides to tokens (integers)

        Args:
            seq::str
                string of DNA nucleotides

        Returns:
            dna_tokens::list
                list containing tokenized DNA nucleotides
        """
        nt_mapping = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
        dna_tokens = []
        for ch in seq:
            dna_tokens.append(nt_mapping[ch])
        return dna_tokens

    def tokenize_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenizes the entire dataset: DNA as well as domains

        Returns:
            (seqs_stack, labs_stack)::tuple
                tuple of PyTorch tensors containing sequences and labels
        """
        num_domains = self.num_domains
        seq_list, lab_list = [], []
        for promseq, domains in self.domains_dict.values():
            prom_tokens = torch.Tensor(self.tokenize_dna(promseq))
            seq_list.append(prom_tokens)
            domains_tokens = torch.zeros(num_domains)
            for domain in domains:
                domain_idx = self.domain_token_dict[domain]
                domains_tokens[domain_idx - 1] = 1
            domains_tokens = torch.Tensor(domains_tokens)
            lab_list.append(domains_tokens)
        seqs_stack = torch.stack((seq_list))
        labs_stack = torch.stack((lab_list))
        return seqs_stack, labs_stack

def domain_occurrence_cutoff(domains_dict, cutoff):
    """Removes domains from the dataset if they occur less than a specified
        number of times

    Args:
        domains_dict::dict
            dictionary containing all processed data:
                {UniProt ID: (promoter sequence: [domains])}

    Returns:
        (domains_dict, num_domains)::tuple
            tuple with dictionary containing all processed data:
                {UniProt ID: (promoter sequence: [domains])} and the total
                number of unique domains
    """
    count_dict = {}
    for uniprot_id, (promseq, domains) in domains_dict.items():
        for domain in domains:
            count_dict.setdefault(domain, [0, []])
            count_dict[domain][0] += 1
            count_dict[domain][1].append(uniprot_id)
    total = 0
    lower = 0
    for domain, (count, uniprot_ids) in count_dict.items():
        total += count
        if count < cutoff:
            lower += count
            for uniprot_id in uniprot_ids:
                if len(domains_dict[uniprot_id][1]) <= 1:
                    # 1 domain or fewer left: delete entire protein
                    del domains_dict[uniprot_id]
                else:
                    # Delete only the particular domain
                    index = domains_dict[uniprot_id][1].index(domain)
                    del domains_dict[uniprot_id][1][index]
    num_datapoints = len(domains_dict)

    print(f'Domains deleted: {lower} out of {total}')
    print(f'Average n domains per promoter: '
          f'{round((total - lower)/num_datapoints, 2)}')
    return domains_dict, num_datapoints


def load_domains(processed_dir):
    """Loads promoter-domain pairs from the directory with processed files

    Args:
        processed_dir::str
            the directory containing the processed files

    Returns:
        domains_dict::dict
            dictionary containing all processed data:
                {UniProt ID: (promoter sequence: [domains])}
    """
    domains_dict = {}
    for file in os.listdir(f'{processed_dir}/domains'):
        with open(f'{processed_dir}/domains/{file}') as fopen:
            curr_upid, curr_promseq, curr_domains = '', '', []
            for line in fopen:
                words = line.split()
                if words[0].startswith('UPID'):
                    curr_upid = words[1]
                if words[0].startswith('PROMSEQ'):
                    curr_promseq = words[1]
                if words[0].startswith('DOMAINS'):
                    curr_domains = words[1:]
                if curr_upid and curr_promseq and curr_domains:
                    domains_dict[curr_upid] = (curr_promseq, curr_domains)
                    curr_upid = ''
                    curr_promseq = ''
                    curr_domains = []
    return domains_dict

def tokenize_domains(domains_dict):
    """Makes a list of all domains, and makes a dict mapping each unique
        domain to a token (int)

    Args:
        domains_dict::dict
            dictionary containing all processed data:
                {UniProt ID: (promoter sequence: [domains])}

    Returns:
        domain_token_dict::dict
            dictionary containing all domains in the dataset with corresponding
                tokens: {domain: token}
        num_domains::int
            the number of domains contained in the dataset
    """
    domain_token_dict = {}
    num_domains = 1
    for promseq, domains in domains_dict.values():
        for domain in domains:
            if domain not in domain_token_dict:
                domain_token_dict[domain] = num_domains
                num_domains += 1
    return domain_token_dict, num_domains - 1  # started counting from 1

def artificial_data(motif_length, num_motifs, motif_chance,
                    prom_length, num_promoters, device):
    """Generates artificial data and loads it into PyTorch datasets for use
        in a neural net.

    Args:
        motif_length::int
            the length of the promoter motifs to generate
        num_motifs::int
            the number of motif-domain pairs to generate
        motif_chance::float
            the probability for any sequence to have each domain
        prom_length::int
            the length of the promoter sequences to generate
        num_promoters::int
            the number of promoters to generate
        device::torch.device object
            the device on which to train the network

    Returns:
        (trainds, testds)::tuple
            tuple of the training and testing dataset
    """
    pairs = domain_pairs(motif_length, num_motifs)
    trainsize = int(num_promoters*0.8)
    testsize = int(num_promoters*0.2)
    train = generate_domain_dataset(motif_chance, pairs, trainsize,
                                    prom_length, motif_length)
    trainds = DomainDataset(train, prom_length, device)
    test = generate_domain_dataset(motif_chance, pairs, testsize,
                                   prom_length, motif_length)
    testds = DomainDataset(test, prom_length, device)

    return trainds, testds

def real_data(processed_dir, prom_length, device, cutoff=10):
    """Loads real promoter-domain pairs into PyTorch datasets

    Args:
        processed_dir::str
            the name of the directory where the processed files are
        prom_length::int
            the length of the promoters used
        device::torch.device object
            the device on which to train the network
        cutoff::int
            the minimum number of times a domain must occur in the dataset
                in order for it to be kept

    Returns:
        (trainds, testds, num_domains)::tuple
            tuple containing the PyTorch datasets and the number of unique
                domains in the dataset.
    """
    dataset = load_domains(processed_dir)
    dataset, numdatapoints = domain_occurrence_cutoff(dataset, cutoff)
    print(f'Total number of promoter sequences left: {numdatapoints}')
    domain_token_dict, num_domains = tokenize_domains(dataset)

    trainsize = int(len(dataset) * 0.8)
    testsize = int(len(dataset) * 0.2)
    keys = list(dataset.keys())
    train = keys[:trainsize]
    test = keys[trainsize:]
    traindict = {k: dataset[k] for k in train}
    testdict = {k: dataset[k] for k in test}
    trainds = DomainDataset(traindict, prom_length, device, domain_token_dict,
                            num_domains)
    testds = DomainDataset(testdict, prom_length, device, domain_token_dict,
                           num_domains)
    return trainds, testds, num_domains

if __name__ == "__main__":
    """Helper module: no main function"""
    pass

