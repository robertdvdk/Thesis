"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
import matplotlib.pyplot as plt
import numpy as np

# Function definitions
def count_lstm_freq(file):
    with open(file, 'r') as fopen:
        i = 0
        selected_lines = []
        for line in fopen:
            if '(1000 100%)' in line:
                i += 1
            if i == 3 and line.startswith('<'):
                selected_lines.append(line.strip().replace(' ', '')[1:].replace('<EOS>', ''))
    # freq_dict = {}
    freq_dict = {'a': 0, 'c': 0, 'g': 0, 't': 0}
    for line in selected_lines:
        for ch in line:
            if ch == 'n':
                continue
            # if ch not in freq_dict:
            #     freq_dict[ch] = 0
            freq_dict[ch] += 1
    proportions = {k: (v / sum(freq_dict.values())) for k, v in freq_dict.items()}
    return proportions

def count_dataset_freq(file):
    with open(file, 'r') as fopen:
        dna = []
        for line in fopen:
            dna.append(line.split()[0].lower())
    freq_dict = {'a': 0, 'c': 0, 'g': 0, 't': 0}
    # freq_dict = {}
    for line in dna:
        for ch in line:
            if ch == 'n':
                continue
            # if ch not in freq_dict:
            #     freq_dict[ch] = 0
            freq_dict[ch] += 1
    proportions = {k: (v / sum(freq_dict.values())) for k, v in freq_dict.items()}
    return proportions

def count_real_dataset_freq(file):
    nuc_dict = {}
    aa_dict = {}
    with open('realdna-aa_short.txt', 'r') as fopen:
        for line in fopen:
            dna, prot = line.strip().split('\t')
            for base in dna:
                if base not in nuc_dict:
                    nuc_dict[base] = 1
                else:
                    nuc_dict[base] += 1
            for aa in prot:
                if aa not in aa_dict:
                    aa_dict[aa] = 1
                else:
                    aa_dict[aa] += 1
    nucs, nucs_counts = [], []
    for k, v in sorted(nuc_dict.items()):
        nucs.append(k)
        nucs_counts.append(v)
    aas, aas_counts = [], []
    for k, v in sorted(aa_dict.items()):
        aas.append(k)
        aas_counts.append(v)
    aas_counts = np.divide(aas_counts, np.sum(aas_counts))
    plt.bar(aas, aas_counts)
    plt.suptitle('Amino acid distribution of the dataset')
    plt.xlabel('Amino acid')
    plt.ylabel('Proportion of total amino acids')
    plt.show()
    nucs_counts = np.divide(nucs_counts, np.sum(nucs_counts))
    plt.bar(nucs, nucs_counts)
    plt.suptitle('Nucleotide distribution of the dataset')
    plt.xlabel('Nucleotide')
    plt.ylabel('Proportion of total nucleotides')
    plt.show()
    print(f"GC content: {nucs_counts[1] + nucs_counts[2]}")

def plot_nucbars_ganlstm():
    art_data_groundtruth_freq = count_dataset_freq('/home/klis004/thesis/seq2seq/gan/tensorflow_data_2ediff.txt')
    art_data_gan_freq = count_dataset_freq('/home/klis004/nbk_lustre/gan/tensorflow/logs/2EDIFF_N/2022.10.13-10h33m08s_lecun/samples/samples_25000')
    art_data_lstmff_freq = count_lstm_freq('/home/klis004/thesis/seq2seq/lstm/2ediff_fake_tff.txt')
    art_data_lstmft_freq = count_lstm_freq('/home/klis004/thesis/seq2seq/lstm/2ediff_fake_tft.txt')
    art_data_lstmtt_freq = count_lstm_freq('/home/klis004/thesis/seq2seq/lstm/2ediff_fake_ttt.txt')

    real_data_groundtruth_freq = count_dataset_freq('/home/klis004/thesis/seq2seq/gan/expressiongan/ExpressionGAN/scripts/REAL_ALL/realdna-aa_short_all.txt')
    real_data_lstmff_freq = count_lstm_freq('/home/klis004/thesis/seq2seq/lstm/tff.txt')
    real_data_lstmft_freq = count_lstm_freq('/home/klis004/thesis/seq2seq/lstm/tft.txt')
    real_data_lstmtt_freq = count_lstm_freq('/home/klis004/thesis/seq2seq/lstm/ttt.txt')
    real_data_gan_freq = count_dataset_freq('/home/klis004/nbk_lustre/gan/tensorflow/logs/REAL_ALL/2022.10.12-18h04m38s_lecun/samples/samples_38000')

    arath_real = count_dataset_freq('/home/klis004/thesis/seq2seq/gan/expressiongan/ExpressionGAN/scripts/REAL_ONLY_ARATH/realdna-aa_short_arath.txt')
    arath_gan = count_dataset_freq('/home/klis004/nbk_lustre/gan/tensorflow/logs/REAL_ONLY_ARATH/2022.10.12-18h15m41s_lecun/samples/samples_115000')

    print(arath_real['c'] + arath_real['g'])
    print(arath_gan['c'] + arath_gan['g'])

    # fig, axs = plt.subplots(1, 2, sharey=True, figsize=(11, 8))
    # labels = ['GT', 'GAN', 'LSTM1', 'LSTM2', 'LSTM3']
    # plotdata = []
    # for (i, j, k, l, m) in zip(art_data_groundtruth_freq.values(),
    #              art_data_gan_freq.values(), art_data_lstmff_freq.values(),
    #              art_data_lstmft_freq.values(), art_data_lstmtt_freq.values()):
    #     plotdata.append([i, j, k, l, m])
    # axs[0].bar(labels, plotdata[0], label='A')
    # axs[0].bar(labels, plotdata[1], bottom=plotdata[0], label='C')
    # axs[0].bar(labels, plotdata[2], bottom=[i + j for (i, j) in zip(plotdata[0], plotdata[1])], label='G')
    # axs[0].bar(labels, plotdata[3], bottom=[i + j + k for (i, j, k) in zip(plotdata[0], plotdata[1], plotdata[2])], label='T')
    # axs[0].tick_params(axis='both', labelsize=14)
    # axs[0].set_title('Nucleotide distribution of artificial data', fontsize=16)

    plt.figure(figsize=(2, 2))
    labels = ['GT', 'LSTM']
    plotdata = []
    for (i, j) in zip(real_data_groundtruth_freq.values(),
                 real_data_lstmtt_freq.values()):
        plotdata.append([i, j])
    plt.bar(labels, plotdata[0], label='A', width=0.4)
    plt.bar(labels, plotdata[1], bottom=plotdata[0], label='C', width=0.4)
    plt.bar(labels, plotdata[2], bottom=[i + j for (i, j) in zip(plotdata[0], plotdata[1])], label='G')
    plt.bar(labels, plotdata[3], bottom=[i + j + k for (i, j, k) in zip(plotdata[0], plotdata[1], plotdata[2])], label='T')
    plt.title('Nucleotide distribution of real data', fontsize=16)
    plt.legend(bbox_to_anchor=(1, 1), fontsize=16)
    plt.tick_params(axis='both', labelsize=14)

    plt.show()

def plot_nucbars_gans():
    real_data_groundtruth_freq = count_dataset_freq('/home/klis004/thesis/seq2seq/gan/expressiongan/ExpressionGAN/scripts/REAL_ONLY_ARATH/test_data.txt')
    data_gan_100 = count_dataset_freq('/home/klis004/nbk_lustre/gan/tensorflow/logs/REAL_ONLY_ARATH/2022.10.12-18h15m41s_lecun/samples/samples_100')
    data_gan_1000 = count_dataset_freq('/home/klis004/nbk_lustre/gan/tensorflow/logs/REAL_ONLY_ARATH/2022.10.12-18h15m41s_lecun/samples/samples_1000')
    data_gan_5000 = count_dataset_freq('/home/klis004/nbk_lustre/gan/tensorflow/logs/REAL_ONLY_ARATH/2022.10.12-18h15m41s_lecun/samples/samples_5000')
    data_gan_10000 = count_dataset_freq('/home/klis004/nbk_lustre/gan/tensorflow/logs/REAL_ONLY_ARATH/2022.10.12-18h15m41s_lecun/samples/samples_10000')
    data_gan_25000 = count_dataset_freq('/home/klis004/nbk_lustre/gan/tensorflow/logs/REAL_ONLY_ARATH/2022.10.12-18h15m41s_lecun/samples/samples_25000')

    fig, axs = plt.subplots(1, 1, sharey=True, figsize=(10, 10))
    labels = ['GT', 'GAN_100', 'GAN_1000', 'GAN_5000', 'GAN_10000', 'GAN_25000']
    plotdata = []
    for (i, j, k, l, m, n) in zip(real_data_groundtruth_freq.values(),
                 data_gan_100.values(), data_gan_1000.values(),
                 data_gan_5000.values(), data_gan_10000.values(),
                 data_gan_25000.values()):
        plotdata.append([i, j, k, l, m, n])
    axs.bar(labels, plotdata[0], label='A')
    axs.bar(labels, plotdata[1], bottom=plotdata[0], label='C')
    axs.bar(labels, plotdata[2], bottom=[i + j for (i, j) in zip(plotdata[0], plotdata[1])], label='G')
    axs.bar(labels, plotdata[3], bottom=[i + j + k for (i, j, k) in zip(plotdata[0], plotdata[1], plotdata[2])], label='T')
    plt.suptitle('Nucleotide distribution of Arabidopsis thaliana promoters and generated promoters', fontsize=14)
    plt.xlabel('Nucleotide', fontsize=14)
    plt.ylabel('Proportion of total nucleotides', fontsize=14)
    plt.savefig('nuc_plot_gans_large.png')



def main():
    plot_nucbars_ganlstm()
    # plot_nucbars_gans()

if __name__ == "__main__":
    main()
