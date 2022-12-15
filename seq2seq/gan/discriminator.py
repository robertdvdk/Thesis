"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import dataloader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random

# Function definitions
def make_dataset():
    fake_loc = '/home/klis004/nbk_lustre/gan/tensorflow/logs/3ediff_lang/2022.09.16-16h13m33s_lecun/samples'
    files = ['samples_100000', 'samples_105000', 'samples_110000', 'samples_115000', 'samples_120000', 'samples_125000', 'samples_130000']
    with open('./input4.txt', 'a') as writeopen:
        for file in files:
            with open(f'{fake_loc}/{file}', 'r') as fopen:
                for line in fopen:
                    writeopen.write(f'{line.strip()} 1\n')
                    # if random.random() < 0.5:
                    #     writeopen.write(f'{line.strip()} 1\n')
                    # else:
                    #     writeopen.write(f'{line.strip()} 0\n')
        real_loc = '/home/klis004/nbk_lustre/gan_input/tensorflow_data_3ediff.txt'
        with open(real_loc, 'r') as fopen:
            for line in fopen:
                writeopen.write(f'{line.strip()} 0 \n')

def load_dataset(inputfile):

    input_dict = {}
    with open(inputfile, 'r') as fopen:
        for line in fopen:
            dna, lab = line.strip().split(' ')
            input_dict[dna] = lab
    return input_dict

class DiscDataset(Dataset):
    """Class designed to work with promoter sequence-protein sequence or
        promoter sequence-protein domain datasets as generated in
        get_datasets/main.py
    """

    def __init__(self, input_dict, device):
        super().__init__()
        self.input_dict = input_dict

        seqs, labs = self.tokenize_dataset()
        seqs = F.one_hot(seqs.to(torch.int64))
        seqs, labs = seqs.to(device), labs.to(device)
        self.balance = 1 / torch.mean(labs).item()
        print(f'proportion 1s in data: {torch.mean(labs).item()}')
        self.data = [(x.float(), y) for x, y in zip(seqs, labs)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_balance(self):
        return self.balance

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

    def tokenize_dataset(self):
        """Tokenizes the entire dataset: DNA as well as domains

        Returns:
            (seqs_stack, labs_stack)::tuple
                tuple of PyTorch tensors containing sequences and labels
        """
        seq_list, lab_list = [], []

        for promseq, domains in self.input_dict.items():
            prom_tokens = torch.Tensor(self.tokenize_dna(promseq))
            seq_list.append(prom_tokens)
            domains = torch.Tensor([int(domains)])

            lab_list.append(domains)
        seqs_stack = torch.stack(seq_list)
        labs_stack = torch.stack(lab_list)
        return seqs_stack, labs_stack


class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)

class Discriminator(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Discriminator, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1d = nn.Conv1d(n_chars, hidden, 1)
        self.linear = nn.Linear(seq_len*hidden, 1)

    def forward(self, input):
        output = input.transpose(1, 2) # (BATCH_SIZE, len(charmap), SEQ_LEN)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.seq_len*self.hidden)
        output = self.linear(output)
        return output

class DiscOwn(nn.Module):
    def __init__(self):
        """
        Deepromoter
        :param para_ker: List of kernel size that will be used
        :param input_shape: Specifies the input shape for model (batch size,
            prom len, number of tokens)
        :param pool_kernel: Pooling parameter after CNN
        :param drop: Dropout parameter
        """
        super(DiscOwn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(5, 128, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AvgPool1d(kernel_size=2, stride=2, padding=0),
            nn.Flatten())

        self.fc = nn.Sequential(
            nn.Linear(6400, 50),
            nn.BatchNorm1d(50), nn.Dropout(), nn.ReLU(),
            nn.Linear(50, 20), nn.BatchNorm1d(20), nn.Dropout(), nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (BATCH_SIZE, len(charmap), SEQ_LEN)
        x = self.conv(x)
        x = self.fc(x)
        return x

def trainer(device, num_epochs=1000, batch_size=64, seq_len=100, hidden=512):

    data, charmap, inv_charmap = dataloader.load_dataset(250, 1000, strip=True, data_dir='/home/klis004/nbk_lustre/gan/tensorflow/logs/3ediff_constanteachtergrond/2022.09.16-14h17m25s_lecun/samples/samples_15000')
    init_epoch = 1
    total_iterations = 1000
    table = np.arange(len(charmap)).reshape(-1, 1)
    one_hot = OneHotEncoder()
    one_hot.fit(table)
    model = Discriminator(len(charmap), seq_len, batch_size, hidden)
    counter = 0
    for epoch in range(num_epochs):
        n_batches = int(len(data) / batch_size)
        for idx in range(n_batches):
            _data = np.array([[charmap[c] for c in l] for l in data[idx * batch_size:(idx + 1) * batch_size]],dtype='int32')
            data_one_hot = one_hot.transform(
                _data.reshape(-1, 1)).toarray().reshape(batch_size, -1, len(charmap))
            real_data = torch.Tensor(data_one_hot).to(device)
            print(real_data)

def init_weights(m):
    """Function to initialize weights of a neural network linear layer

    Args:
        m::nn.Module
            a PyTorch neural net module to apply weight initialization to
    Returns:
        None
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def train_function(net, dataset_train, dataset_test, lr, epochs, device,
                   batch_size, DeePromoter=False, verbose=True, pos_weight=1):
    """Function to train a neural network

    Args:
        net::nn.Module
            a PyTorch neural network (nn.Sequential or a subclass of nn.Module)
        dataset_train::torch.utils.data.Dataset
            a tensor training dataset with one-hot encoded inputs and outputs
        dataset_test::torch.utils.data.Dataset
            a tensor testing dataset with one-hot encoded inputs and outputs
        lr::float
            the learning rate of the neural net
        epochs::int
            the number of epochs to use
        device::torch.device object
            the device on which to train the network
        batch_size::int
            the batch size to use while training
        DeePromoter::bool
            whether or not the used network is DeePromoter
        verbose::bool
            whether or not to print accuracy stats every 5 epochs

    Returns:
        (trainaccs, testaccs, precisions, recalls, specificities)::tuple
            Tuple of lists where each list contains accuracy parameters as
            measured once every 5 epochs
        """
    net = net.to(device)
    print(f'training on: {device}')
    net.apply(init_weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    trainloader = DataLoader(dataset_train, batch_size=batch_size,
                             shuffle=True, collate_fn=None)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True,
                            collate_fn=None)
    trainaccs, testaccs, precisions, recalls, specificities = [], [], [], [], \
                                                              []
    for i in range(epochs):
        print(f'Epoch: {i}')
        for x, y in trainloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            # DeePromoter permutes x inside the net
            # if DeePromoter:
            #     y_pred = net(x)
            # else:
            #     y_pred = net(x.permute(0, 2, 1))
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            trainpreds = torch.where(y_pred < 0, 0, 1)
            trainaccuracy = round(torch.mean(
                torch.where(trainpreds == y, 1, 0).to(torch.float)).item(), 2)

        if i % 5 == 0:
            xtest, ytest = next(iter(testloader))
            print(f'Training loss: {loss}')
            # For running on network other than DeePromoter
            # if DeePromoter:
            #     ytest_pred = net(xtest)
            # else:
            #     ytest_pred = net(xtest.permute(0, 2, 1))
            ytest_pred = net(xtest)

            testpreds = torch.where(ytest_pred < 0, 0, 1)
            testaccuracy = torch.mean(
                torch.where(testpreds == ytest, 1, 0).to(torch.float)).item()
            testaccuracy = round(testaccuracy, 3)
            trainaccs.append(trainaccuracy)
            testaccs.append(testaccuracy)
            if verbose:
                print(f'Test accuracy: {testaccuracy}')
    return trainaccs, testaccs, precisions, recalls, specificities

def real_data(device):
    """Loads real promoter-domain pairs into PyTorch datasets

    Args:
        processed_dir::str
            the name of the directory where the processed files are
        device::torch.device object
            the device on which to train the network
        begin_base::int
            the base at which to begin (1000 is Transcription Start Site)
        end_base::int
            the base at which to end (1000 is Transcription Start Site)
        cutoff::int
            the minimum number of times a domain must occur in the dataset
                in order for it to be kept

    Returns:
        (trainds, testds, num_domains)::tuple
            tuple containing the PyTorch datasets and the number of unique
                domains in the dataset.
    """
    dataset = load_dataset('./input2.txt')
    trainsize = int(len(dataset) * 0.8)
    keys = list(dataset.keys())
    random.shuffle(keys)
    train = keys[:trainsize]
    test = keys[trainsize:]
    traindict = {k: dataset[k] for k in train}
    testdict = {k: dataset[k] for k in test}
    trainds = DiscDataset(traindict, device)
    testds = DiscDataset(testdict, device)
    return trainds, testds

def main():
    # make_dataset()
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    trainds, testds = real_data(device)
    # model = DiscOwn()
    model = Discriminator(5, 100, 64, 128)
    train_function(model, trainds, testds, 0.001, 100, device, 64)

    # trainer(device)

if __name__ == "__main__":
    main()
