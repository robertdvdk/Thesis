"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Function definitions

def count_sig_motifs(file):
    vals = []
    sig_motifs = []
    sig_motifs_counts = {}
    sig_motifs_countsnew = []
    with open(file, 'r') as fopen:
        for line in fopen.readlines()[1:]:
            if not line.startswith('M'):
                continue
            words = line.split('\t')
            if float(words[-2]) < 0.05:
                # sig_motifs_countsnew.append(words[0])
                if words[0] not in sig_motifs_counts:
                    sig_motifs_counts[words[0]] = 0
                sig_motifs_counts[words[0]] += 1
            vals.append(float(words[-2]))
    return vals, sig_motifs_counts


def plot_motifshist():
    gt_motifs, sig_motifs_gt = count_sig_motifs('REAL_ONLY_ARATH/fimo_gt.tsv')
    gan_motifs_100, sig_motifs_100 = count_sig_motifs('REAL_ONLY_ARATH/fimo_100.tsv')
    gan_motifs_1000, sig_motifs_1000 = count_sig_motifs('REAL_ONLY_ARATH/fimo_1000.tsv')
    gan_motifs_5000, sig_motifs_5000 = count_sig_motifs('REAL_ONLY_ARATH/fimo_5000.tsv')
    gan_motifs_10000, sig_motifs_10000 = count_sig_motifs('REAL_ONLY_ARATH/fimo_10000.tsv')
    gan_motifs_25000, sig_motifs_25000 = count_sig_motifs('REAL_ONLY_ARATH/fimo_25000.tsv')
    # print(len(sig_motifs_gt), len(sig_motifs_100), len(sig_motifs_1000), len(sig_motifs_5000), len(sig_motifs_10000), len(sig_motifs_25000))
    # print(sorted(sig_motifs_gt.values(), reverse=True))
    # print(sorted(sig_motifs_100.values(), reverse=True))
    # print(sorted(sig_motifs_1000.values(), reverse=True))
    # print(sorted(sig_motifs_5000.values(), reverse=True))
    # print(sorted(sig_motifs_10000.values(), reverse=True))
    # print(sorted(sig_motifs_25000.values(), reverse=True))

    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    fig.text(0.5, 0.02, 'Motif frequency', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Count', va='center', rotation='vertical', fontsize=14)
    axs[0].hist(sig_motifs_gt.values(), cumulative=True)
    axs[0].set_title('Ground truth', fontsize=14)
    axs[1].hist(sig_motifs_5000.values(), bins=100, cumulative=True)
    axs[1].axvline(1000, color='red')
    axs[1].set_title('GAN 5000 iterations', fontsize=14)
    axs[2].hist(sig_motifs_25000.values(), cumulative=True)
    axs[2].set_title('GAN 25000 iterations', fontsize=14)

    plt.savefig('cumulative_hist_large.png')
    print(sorted(sig_motifs_5000.values(), reverse=True))
    print(sorted(sig_motifs_25000.values(), reverse=True))
    print(sorted(sig_motifs_gt.values(), reverse=True))
    print([i for i in sig_motifs_5000.values() if i > 1000])

    # fig, axs = plt.subplots(2, 3, figsize=(12, 12))
    fig, axs = plt.subplots(1, 2, figsize=(8, 8), sharey=True)
    bins = [i/20 for i in range(0, 20)]
    fig.text(0.5, 0.02, 'q value', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Count', va='center', rotation='vertical', fontsize=14)
    axs[0].hist(gt_motifs, bins=bins)
    axs[0].set_title('Ground truth', fontsize=14)
    # axs[0, 1].hist(gan_motifs_100, bins=bins)
    # axs[0, 1].set_title('GAN 100 iterations', fontsize=14)
    # axs[0, 2].hist(gan_motifs_1000, bins=bins)
    # axs[0, 2].set_title('GAN 1000 iterations', fontsize=14)
    # axs[1, 0].hist(gan_motifs_5000, bins=bins)
    # axs[1, 0].set_title('GAN 5000 iterations', fontsize=14)
    # axs[1, 1].hist(gan_motifs_10000, bins=bins)
    # axs[1, 1].set_title('GAN 10000 iterations', fontsize=14)
    # axs[1, 2].hist(gan_motifs_25000, bins=bins)
    # axs[1, 2].set_title('GAN 25000 iterations', fontsize=14)
    axs[1].hist(gan_motifs_25000, bins=bins)
    axs[1].set_title('GAN', fontsize=14)
    fig.suptitle('q value distribution of motifs', fontsize=14)
    # plt.savefig('motif_hist_large')
    plt.show()
def hamming_dist(file):
    sim_matrix = [[0 for i in range(640)] for i in range(640)]
    lines = []
    with open(file, 'r') as fopen:
        for line in fopen:
            if not line.startswith('>'):
                lines.append(line)
    for i in range(640):
        for j in range(640):
            for ch in range(100):
                if lines[i][ch] == lines[j][ch]:
                    sim_matrix[i][j] += 1
    X = np.array(sim_matrix)
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
    labels = [i for i in range(640)]
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
    # for i in labels:
    #     plt.annotate(i, (X_embedded[i, 0], X_embedded[i, 1]))
    plt.show()


def main():
    plot_motifshist()
    # hamming_dist('/Users/robertvdklis/Documents/code/Thesis/gan_outputs/logs/REAL_ONLY_ARATH/samples_5000_fasta.txt')
    # hamming_dist('/Users/robertvdklis/Documents/code/Thesis/gan_outputs/logs/REAL_ONLY_ARATH/samples_25000_fasta.txt')
    # hamming_dist('/Users/robertvdklis/Documents/code/Thesis/gan_outputs/logs/REAL_ONLY_ARATH/2022.10.12-18h15m41s_lecun/samples/test_data.txt')
if __name__ == "__main__":
    main()
