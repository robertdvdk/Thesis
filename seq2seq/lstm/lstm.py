"""
Author: Robert van der Klis

Learns an LSTM-LSTM model to predict protein sequences based on
    promoter sequences

Usage: python3 lstm.py
Adapted from
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""


# Import statements
from __future__ import unicode_literals, print_function, division
from io import open
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Function definitions
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 250
K = 3

class Lang:
    """Finds all nucleotides/amino acids present in the data set and adds them
        to a class

    Args:
        name::string
            the name of the language to add
    """
    def __init__(self, name):
        self.name = name
        self.letter2index = {}
        self.letter2count = {}
        self.index2letter = {0: "<SOS>", 1: "<EOS>"}
        self.n_letters = 2  # Count SOS and EOS

    def addLine(self, line):
        # for i in range(len(line) - K):
        #     self.addKmer(line[i:i+K])

        for letter in list(line):
            self.addLetter(letter)

    def addLetter(self, letter):
        if letter not in self.letter2index:
            self.letter2index[letter] = self.n_letters
            self.letter2count[letter] = 1
            self.index2letter[self.n_letters] = letter
            self.n_letters += 1
        else:
            self.letter2count[letter] += 1

    def addKmer(self, kmer):
        if kmer not in self.letter2index:
            self.letter2index[kmer] = self.n_letters
            self.letter2count[kmer] = 1
            self.index2letter[self.n_letters] = kmer
            self.n_letters += 1
        else:
            self.letter2count[kmer] += 1


def normalizeString(s):
    """Converts lines to lower cases and strips off whitespaces

    Args:
        s::string
            the string to be processed

    Returns:
        s::string
            the processed string
    """
    return s.lower().strip()

def readLangs(lang1, lang2, reverse=False):
    """Reads in a file, and converts it to pairs of lines in both languages

    Args:
        lang1::str
            the name of the first language
        lang2::str
            the name of the second language
        reverse::bool
            if true, the lines in the second language are the input. if false,
            the lines in the first language are the input

    Returns:
        (input_lang, output_lang, pairs)::tuple
            tuple of input and output Lang objects, and the sentence pairs
    """
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('/home/klis004/thesis/seq2seq/lstm/diff2%s-%s_short.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # lines = open('test.txt', encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    # pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [[normalizeString(s) for s in l.split()] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2, reverse=False):
    """Prepares data for processing by LSTM

    Args:
        lang1::str
            the name of the first language
        lang2::str
            the name of the second language
        reverse::bool
            if true, the lines in the second language are the input. if false,
            the lines in the first language are the input

    Returns:
        (input_lang, output_lang, pairs)::tuple
            tuple of input and output Lang objects, and the sentence pairs
    """
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addLine(pair[0])
        output_lang.addLine(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_letters)
    print(output_lang.name, output_lang.n_letters)
    return input_lang, output_lang, pairs




class EncoderRNN(nn.Module):
    """The encoder part of the network
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, a):
        (hidden, cell) = a
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        return output, (hidden, cell)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def initCell(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    """The decoder part of the network
    """
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, a):
        (hidden, cell) = a
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = self.out(output[0])
        return output, (hidden, cell)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def initCell(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, a, encoder_outputs):
        (hidden, cell) = a
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0],
                                                      hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))

        output = self.out(output[0])

        return output, (hidden, cell), attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def initCell(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromLine(lang, line):
    """Returns the index corresponding to a particular letter in the Lang
        object

    Args:
        lang::Lang object
            Lang object of the language in which to look for the index
        line::str
            string of which characters will be looked up in the Lang object

    Returns:
        indices::list
            list of indices, where each index corresponds to that character's
                index in the Lang object
    """
    # indices = []
    # for i in range(len(line) - K):
    #     kmer = line[i:i+K]
    #     indices.append(lang.letter2index[kmer])
    # return indices


    return [lang.letter2index[letter] for letter in list(line)]


def tensorFromLine(lang, line):
    """Returns a line of characters as tensor of their indices in the Lang
        object

    Args:
        lang::Lang object
            Lang object of the language in which to look for the index
        line::str
            string of which characters will be looked up in the Lang object

    Returns:
        indices::Tensor
            Tensor of indices, where each index corresponds to that character's
                index in the Lang object
    """
    indexes = indexesFromLine(lang, line)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    """Converts both input and output line to tensors and returns tensors as
        pair

    Args:
        pair::list
            list of strings, where first string is string in input language,
                and second string is string in output language

    Returns:
        pair::tuple
            tuple of Tensors, where first tensor is tensor of indices in input
                lang, and second tensor is tensor of indices in output lang
        """
    input_tensor = tensorFromLine(input_lang, pair[0])
    target_tensor = tensorFromLine(output_lang, pair[1])
    # return (F.one_hot(input_tensor), F.one_hot(target_tensor))
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, attn=False):
    """Trains encoder-decoder network on dataset

    Args:
        input_tensor::Tensor
            Tensor of indices in input lang
        target_tensor::Tensor
            Tensor of indices in output lang
        encoder::nn.Module
            the encoder to use
        decoder::nn.Module
            the decoder to use
        encoder_optimizer::torch.optim
            optimization procedure to use for the encoder
        decoder_optimizer::torch.optim
            optimization procedure to use for the decoder
        criterion::loss_fn
            the loss function to use in training
        max_length::int
            the maximum length of any sequence

    Returns:
        loss::float
            the average loss per character of the sequence
    """
    encoder_hidden = encoder.initHidden()
    encoder_cell = encoder.initCell()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, (encoder_hidden, encoder_cell) = encoder(input_tensor[ei], (encoder_hidden, encoder_cell))
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if not attn:
                decoder_output, (decoder_hidden, decoder_cell) = decoder(
                    decoder_input, (decoder_hidden, decoder_cell))
            else:
                decoder_output, (decoder_hidden, decoder_cell), decoder_attention = decoder(
                    decoder_input, (decoder_hidden, decoder_cell), encoder_outputs)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if not attn:
                decoder_output, (decoder_hidden, decoder_cell) = decoder(
                    decoder_input, (decoder_hidden, decoder_cell))
            else:
                decoder_output, (decoder_hidden, decoder_cell), decoder_attention = decoder(
                    decoder_input, (decoder_hidden, decoder_cell), encoder_outputs)

            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math


def asMinutes(s):
    """Calculates how many minutes a given number of seconds is

    Args:
        s::float
            number of seconds

    Returns:
        m::float
            number of minutes
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    """Calculates the difference between two points in time and returns the
        amount of time left to complete the job

    Args:
        since::float
            the first time point
        percent::float
            the percentage of work that has been done

    Returns:
        left::float
            the amount of time that is still left to complete job
    """
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, input_lang, output_lang, pairs,
               print_every=10, plot_every=100000, learning_rate=0.001,
               attn=False, FACE=False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang)
                      for i in range(n_iters)]

    if FACE:
        each_letter_count = torch.FloatTensor(list(output_lang.letter2count.values()))
        total_letter_count = torch.sum(torch.FloatTensor(list(output_lang.letter2count.values())))
        each_letter_weight = torch.divide(total_letter_count, each_letter_count)
        newlines = len(pairs)
        newline_weight = torch.FloatTensor([total_letter_count / newlines])

        # Weights of [<SOS>, <EOS>, ... amino acids]
        weights = torch.cat([torch.ones((1)), newline_weight, each_letter_weight]).to(device)

        criterion = nn.CrossEntropyLoss(weight=weights, reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')


    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, attn=attn)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            evaluateRandomly(encoder, decoder, input_lang, output_lang,
                             pairs, n=2, attn=attn)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence, input_lang, output_lang,
             max_length=MAX_LENGTH, attn=False):
    with torch.no_grad():
        input_tensor = tensorFromLine(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_cell = encoder.initCell()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, (encoder_hidden, encoder_cell) = \
                encoder(input_tensor[ei], (encoder_hidden, encoder_cell))
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        decoded_words = []

        for di in range(max_length):
            if not attn:
                decoder_output, (decoder_hidden, decoder_cell) = decoder(
                    decoder_input, (decoder_hidden, decoder_cell))
            else:
                decoder_output, (decoder_hidden, decoder_cell), decoder_attention = decoder(
                    decoder_input, (decoder_hidden, decoder_cell), encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2letter[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, n=10,
                     attn=False):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0], input_lang,
                                output_lang, attn=attn)
        output_sentence = ''.join(output_words)
        print('<', output_sentence)
        print('')

def training_loop(hidden_sizes, attn, FACE, reverse):
    input_lang, output_lang, pairs = prepareData('dna', 'aa', reverse=reverse)
    direction = 'aa-dna' if reverse else 'dna-aa'

    for hidden_size in hidden_sizes:
        print(f'hidden size: {hidden_size}, attn: {attn}, FACE: {FACE}, '
              f'direction: {direction}')
        enc1 = EncoderRNN(input_lang.n_letters, hidden_size).to(device)
        if attn:

            dec1 = AttnDecoderRNN(hidden_size, output_lang.n_letters).to(device)
        else:
            dec1 = DecoderRNN(hidden_size, output_lang.n_letters).to(device)
        trainIters(enc1, dec1, 1000, input_lang, output_lang, pairs,
                   attn=attn, FACE=FACE)
        evaluateRandomly(enc1, dec1, input_lang, output_lang, pairs, attn=attn)

def main():


    hidden_sizes = [128, 512, 2048]
    # prom->prot, no attn, no FACE
    # reverse = False
    # attn = False
    # FACE = False
    # training_loop(hidden_sizes, attn, FACE, reverse=reverse)

    # # prom->prot, no attn, with FACE
    # reverse = False
    # attn = False
    # FACE = True
    # training_loop(hidden_sizes, attn, FACE, reverse=reverse)
    # #
    # #
    # prom->prot, with attn, with FACE
    # reverse = False
    # attn = True
    # FACE = True
    # training_loop(hidden_sizes, attn, FACE, reverse=reverse)
    #
    # # prot->prom, no attn, no FACE
    # reverse = True
    # attn = False
    # FACE = False
    # training_loop(hidden_sizes, attn, FACE, reverse=reverse)
    #
    # # prot->prom, no attn, with FACE
    reverse = True
    attn = False
    FACE = True
    training_loop(hidden_sizes, attn, FACE, reverse=reverse)
    #
    # # prot-> prom, with attn, with FACE
    # reverse = True
    # attn = True
    # FACE = True
    # training_loop(hidden_sizes, attn, FACE, reverse=reverse)

    # for hidden_size in hidden_sizes:
    #     print(f'hidden size: {hidden_size}, attn: {attn}, FACE: {FACE}')
    #     enc1 = EncoderRNN(input_lang.n_letters, hidden_size).to(device)
    #     dec1 = DecoderRNN(hidden_size, output_lang.n_letters).to(device)
    #     trainIters(enc1, dec1, 1000, attn=attn, FACE=FACE)
    #     evaluateRandomly(enc1, dec1, attn=attn)

    # # no attn, with FACE
    # attn = False
    # FACE = True
    # for hidden_size in hidden_sizes:
    #     print(f'hidden size: {hidden_size}, attn')
    #     enc1 = EncoderRNN(input_lang.n_letters, hidden_size).to(device)
    #     dec1 = DecoderRNN(hidden_size, output_lang.n_letters).to(device)
    #     trainIters(enc1, dec1, 1000, attn=attn, FACE=FACE)
    #     evaluateRandomly(enc1, dec1, attn=attn)
    #
    # # with attn, with FACE
    # attn = True
    # FACE = True
    # for hidden_size in hidden_sizes:
    #     print(f'hidden size: {hidden_size}, attn')
    #     enc1 = EncoderRNN(input_lang.n_letters, hidden_size).to(device)
    #     dec1 = DecoderRNN(hidden_size, output_lang.n_letters).to(device)
    #     trainIters(enc1, dec1, 1000, attn=attn, FACE=FACE)
    #     evaluateRandomly(enc1, dec1, attn=attn)

if __name__ == "__main__":
    main()
