"""
Author: Robert van der Klis

Counts number of approximate matches for a given motif. Used to quantify
performance of GANs.
"""


# Import statements

# Function definitions
def read_data(data_loc):
    with open(data_loc, 'r') as fopen:
        data = fopen.read().strip().lower()
    return data


def match_pattern(pattern, data, threshold=0.8):
    approx_matches = 0
    for i in range(len(data) - len(pattern) + 1):
        window = data[i:i+len(pattern)]
        matches = 0
        for nuc_data, nuc_pattern in zip(window, pattern):
            if nuc_data == nuc_pattern:
                matches += 1
        if matches / len(pattern) >= threshold:
            approx_matches += 1
    lines = len(data)/100
    return approx_matches, approx_matches/lines

def read_lstm(data_loc):
    data = ''
    started = 0
    lines = 0
    with open(data_loc, 'r') as fopen:
        for line in fopen:
            if "100%" in line:
                started = 1
            if started:
                if line.startswith('<'):
                    data += line.replace(' ', '')[1:]
                    lines += 1
            if lines >= 12:
                started = 0
                lines = 0
    return data


def main():
    pattern = 'CAGATTTTCA'.lower()
    threshold = 1.0
    datasets_tensorflow = ['/home/klis004/nbk_lustre/gan/tensorflow/logs/1ediff/2022.09.14-12h20m36s_lecun/samples/samples_1400', '/home/klis004/nbk_lustre/gan/tensorflow/logs/2ediff/2022.09.13-12h08m37s_lecun/samples/samples_2500', '/home/klis004/nbk_lustre/gan/tensorflow/logs/3ediff_constanteachtergrond/2022.09.16-14h17m25s_lecun/samples/samples_15000', '/home/klis004/nbk_lustre/gan/tensorflow/logs/3ediff/2022.09.13-12h24m23s_lecun/samples/samples_7000', '/home/klis004/nbk_lustre/gan/tensorflow/logs/3ediff_lang/2022.09.16-16h13m33s_lecun/samples/samples_135000']
    for data_loc in datasets_tensorflow:
        data = read_data(data_loc)
        approx_matches = match_pattern(pattern, data, threshold=threshold)
        print(approx_matches)

    datasets_oud = ['/home/klis004/nbk_lustre/gan/samples/repeatmotif/sampled_146.txt', '/home/klis004/nbk_lustre/gan/samples/motif_1x_echtvasteplek/sampled_25.txt',  '/home/klis004/nbk_lustre/gan/samples/motif_1x_vasteplek/sampled_30.txt']
    for data_loc in datasets_oud:
        data = read_data(data_loc)
        approx_matches = match_pattern(pattern, data, threshold=threshold)
        print(approx_matches)

    datasets_lstm = ['/home/klis004/thesis/seq2seq/lstm/1ediff_fake_tff.txt', '/home/klis004/thesis/seq2seq/lstm/2ediff_fake_tff.txt', '/home/klis004/thesis/seq2seq/lstm/2ediff_fake_ttt.txt']
    for data_loc in datasets_lstm:
        data = read_lstm(data_loc)
        approx_matches = match_pattern(pattern, data, threshold=threshold)
        print(approx_matches)

if __name__ == "__main__":
    main()
