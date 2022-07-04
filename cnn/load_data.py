"""
Loads real and artificial data into datasets PyTorch can work with
"""

# Import statements
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import sys
from random import seed
import artificial_dataset

# Function definitions
class DomainDataset(Dataset):
    """Class designed to work with datasets as generated in
    get_datasets/main.py
    """
    def __init__(self, domains_dict, prom_length):
        super().__init__()
        self.domains_dict = domains_dict
        self.prom_length = prom_length
        seqs, labs = self.tokenize_dataset()
        seqs = F.one_hot(seqs.to(torch.int64))
        self.data = [(x.float(), y) for x, y in zip(seqs, labs)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def tokenize_domains(self):
        """Makes a list of all domains, and makes a dict mapping each unique
         domain to a token (int)

        Returns: dict of {domain: token}
        """
        domain_token_dict = {}
        num_domains = 1
        for promseq, domains in self.domains_dict.values():
            for domain in domains:
                if domain not in domain_token_dict:
                    domain_token_dict[domain] = num_domains
                    num_domains += 1
        return domain_token_dict, num_domains

    def tokenize_dna(self, seq):

        nt_mapping = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
        dna_tokens = []
        for ch in seq:
            dna_tokens.append(nt_mapping[ch])
        return dna_tokens

    def tokenize_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        domain_token_dict, num_domains = self.tokenize_domains()
        num_domains = len(list(domain_token_dict.keys()))
        seq_list, lab_list = [], []
        for promseq, domains in self.domains_dict.values():
            prom_tokens = torch.Tensor(self.tokenize_dna(promseq))
            seq_list.append(prom_tokens)
            domains_tokens = torch.zeros(num_domains)
            for domain in domains:
                domain_idx = domain_token_dict[domain]
                domains_tokens[domain_idx - 1] = 1
            domains_tokens = torch.Tensor(domains_tokens)
            lab_list.append(domains_tokens)
        seqs_stack = torch.stack((seq_list))
        labs_stack = torch.stack((lab_list))
        return seqs_stack, labs_stack

def domain_occurrence_cutoff(domains_dict, cutoff):
    count_dict = {}
    for uniprot_id, (promseq, domains) in domains_dict.items():
        for domain in domains:
            count_dict.setdefault(domain, [0, uniprot_id])
            count_dict[domain][0] += 1
    total = 0
    lower = 0
    for domain, (count, uniprot_id) in count_dict.items():
        total += count
        if count < cutoff and uniprot_id in domains_dict:
            lower += count
            if len(domains_dict[uniprot_id][1]) <= 1:
                # 1 domain or fewer left: delete entire protein
                del domains_dict[uniprot_id]
            else:
                # Delete only the particular domain
                index = domains_dict[uniprot_id][1].index(domain)
                del domains_dict[uniprot_id][1][index]
    print(f'domains deleted: {lower} out of {total}')
    return domains_dict


def load_domains(processed_dir):
    domains_dict = {}
    for file in os.listdir(f'{processed_dir}/domains_old'):
        with open(f'{processed_dir}/domains_old/MACMU.txt') as fopen:
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

def load_into_tensor(domains_dict, prom_length, num_domains):
    seq_tensor = torch.zeros(0, prom_length, 5)
    lab_tensor = torch.zeros(0, num_domains)
    for uniprot_id, (promseq, domains) in domains_dict.items():
        promseq_tokenized = tokenize_dna(promseq)
        promseq_tokenized = torch.Tensor(promseq_tokenized)
        promseq_onehot = F.one_hot(promseq_tokenized.to(torch.int64))
        seq_tensor = torch.cat(tensors=(seq_tensor, promseq_onehot.reshape(1,
                                                            prom_length, 5)))
    return seq_tensor

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def train_function(net, dataset_train, dataset_test, lr, epochs):
    net.apply(init_weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    dataloader = DataLoader(dataset_train, batch_size=100, shuffle=True,
                            collate_fn=None)
    testloader = DataLoader(dataset_test, batch_size=100, shuffle=True,
                            collate_fn=None)
    for _ in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            y_pred = net(x.permute(0, 2, 1))
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            trainpreds = torch.where(y_pred < 0, 0, 1)
            train_accuracy = round(torch.mean(
                torch.where(trainpreds == y, 1, 0).to(torch.float)).item(), 2)
        print(f'train acc: {train_accuracy}')
        print(f'BCEWithLogitsLoss: {loss}')
        xtest, ytest = next(iter(testloader))
        # print(ytest)
        ytest_pred = net(xtest.permute(0, 2, 1))
        testpreds = torch.where(ytest_pred < 0, 0, 1)
        accuracy = round(torch.mean(
            torch.where(testpreds == ytest, 1, 0).to(torch.float)).item(), 2)
        TP = torch.sum(
            torch.where(((testpreds == 1) & (ytest == 1)), 1, 0)).item()
        FP = torch.sum(
            torch.where(((testpreds == 1) & (ytest == 0)), 1, 0)).item()
        FN = torch.sum(
            torch.where(((testpreds == 0) & (ytest == 1)), 1, 0)).item()
        if TP + FP != 0:
            precision = round(TP / (TP + FP), 2)
        else:
            precision = 'no TP and no FP'
        if TP + FN != 0:
            recall = round(TP / (TP + FN), 2)
        else:
            recall = 'no TP and no FN'
        print(f'test acc: {accuracy}')
        print(f'precision: {precision}')
        print(f'recall: {recall}')

if __name__ == "__main__":
    """The main function of this module"""
    # domains_dict = load_domains('/home/klis004/nbk_lustre/processed_data')
    # domains_dict = domain_occurrence_cutoff(domains_dict, 10)

    seed(1)
    motif_length = 10
    prom_length = 100
    pairs = artificial_dataset.domain_pairs(motif_length, 10)

    train = artificial_dataset.generate_domain_dataset(0.01, pairs, 100000,
                                                    prom_length, motif_length)
    trainds = DomainDataset(train, prom_length)
    test = artificial_dataset.generate_domain_dataset(0.01, pairs, 500,
                                                    prom_length, motif_length)
    testds = DomainDataset(test, prom_length)
    net = nn.Sequential(
        nn.Conv1d(5, 5, 11, 1, padding='same'), nn.BatchNorm1d(5), nn.ReLU(),

        nn.Flatten(),
        nn.Linear(500, 50), nn.BatchNorm1d(50), nn.Dropout(0.5), nn.ReLU(),
        nn.Linear(50, 10))
    train_function(net, trainds, testds, 1e-1, 1000)

