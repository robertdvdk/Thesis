"""
Trains neural network on artificial and real promoter-domain data

Author: Robert van der Klis

Usage: python3 train.py
"""
# Import statements
import torch
import torch.nn as nn
from load_data import artificial_data, real_data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from DeePromoter import DeePromoter

# Function definitions
class BCEWithLogitsLossWeighted(torch.nn.Module):
    """From: https://stackoverflow.com/questions/57021620/
    how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch"""
    def __init__(self, weight, *args, **kwargs):
        super().__init__()
        # Notice none reduction
        self.bce = torch.nn.BCEWithLogitsLoss(*args, **kwargs, reduction="none")
        self.weight = weight

    def forward(self, logits, labels):
        loss = self.bce(logits, labels)
        binary_labels = labels.bool()
        loss[binary_labels] *= labels[binary_labels] * self.weight
        # Or any other reduction
        return torch.mean(loss)

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
    criterion = BCEWithLogitsLossWeighted(pos_weight)
    # criterion = nn.BCEWithLogitsLoss()
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
            if DeePromoter:
                y_pred = net(x)
            else:
                y_pred = net(x.permute(0, 2, 1))

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
            if DeePromoter:
                ytest_pred = net(xtest)
            else:
                ytest_pred = net(xtest.permute(0, 2, 1))

            testpreds = torch.where(ytest_pred < 0, 0, 1)
            testaccuracy = torch.mean(
                torch.where(testpreds == ytest, 1, 0).to(torch.float)).item()
            testaccuracy = round(testaccuracy, 3)

            TP = torch.sum(
                torch.where(((testpreds == 1) & (ytest == 1)), 1, 0)).item()
            FP = torch.sum(
                torch.where(((testpreds == 1) & (ytest == 0)), 1, 0)).item()
            FN = torch.sum(
                torch.where(((testpreds == 0) & (ytest == 1)), 1, 0)).item()
            TN = torch.sum(
                torch.where(((testpreds == 0) & (ytest == 0)), 1, 0)).item()
            if TP + FP != 0:
                precision = round(TP / (TP + FP), 3)
            else:
                precision = 0
            if TP + FN != 0:
                recall = round(TP / (TP + FN), 3)
            else:
                recall = 0
            if TN + FP != 0:
                specificity = round(TN / (TN + FP), 3)
            else:
                specificity = 0
            trainaccs.append(trainaccuracy)
            testaccs.append(testaccuracy)
            precisions.append(precision)
            recalls.append(recall)
            specificities.append(specificity)
            if verbose:
                print(f'Test accuracy: {testaccuracy}\nPrecision: {precision}'
                      f'\nRecall: {recall}\nSpecificity: {specificity}')
    return trainaccs, testaccs, precisions, recalls, specificities

def plot_accuracy(accuracy_stats, epochs, save_as_pdf=False):
    """Plots accuracy stats, either displays plots or saves them to pdf

    Args:
        accuracy_stats::tuple
            contains lists of training accuracy, testing accuracy, precision,
                recall, specificity of trained model
        epochs::int
            the number of epochs the model was trained for
        save_as_pdf::bool
            whether or not to save the plot to a pdf

    Returns:
        None
    """
    with PdfPages('rawoutput_smallnetwork.pdf') as pdf:
        xaxis = list(range(0, epochs, 5))
        for i, (trainaccs, testaccs, precisions, recalls, specificities) in \
            enumerate(accuracy_stats):
            plt.figure()
            plt.plot(xaxis, trainaccs, label='Train accuracy')
            plt.plot(xaxis, testaccs, label='Test accuracy')
            plt.plot(xaxis, precisions, label='Precision')
            plt.plot(xaxis, recalls, label='Recall')
            plt.plot(xaxis, specificities, label='Specificity')
            plt.ylim(0, 1)
            plt.xlabel('Epochs')
            plt.legend()
            if save_as_pdf:
                pdf.savefig()
                plt.close()
            else:
                plt.show()

def main(artificial=True):
    """Main function that trains the model

    Args:
        domain::bool
            whether the dataset to train on is promoter-protein domain or
            promoter-protein sequence
    """
    if torch.cuda.is_available():
        dev = "cuda:1"
    else:
        dev = "cpu"
    device = torch.device(dev)
    processed_dir = '/home/klis004/nbk_lustre/processed_data'
    all_accuracies = []

    if artificial:
        epochs = 250
        batch_size = 64
        motif_chance = 0.3
        domain_pairs = 2
        num_sequences = 50000
        prom_length = 100
        motif_length = 10

        for domain_pairs in [1, 2, 3, 4, 5, 10]:
            motif_chance = 0.3
            num_sequences = 50000
            prom_length = 100
            motif_length = 10
            trainds, testds = artificial_data(motif_length, domain_pairs,
                                              motif_chance, prom_length,
                                              num_sequences, device)
            net = DeePromoter(para_ker=[27, 14, 7], total_domains=domain_pairs,
                              input_shape=(64, prom_length, 5))
            print(f"#######TRAINING########\ndomain pairs: {domain_pairs}\t"
                  f"motif_chance: {motif_chance}\tnum_sequences: {num_sequences}\t"
                  f"prom_length: {prom_length}\tmotif_length: {motif_length}\n")
            all_accuracies.append(train_function(net, trainds, testds, 1e-2,
                                                 epochs, device, 64,
                                                 DeePromoter=True, verbose=True))

        for motif_chance in [0.1, 0.3, 0.5, 0.7, 0.9]:
            domain_pairs = 2
            num_sequences = 50000
            prom_length = 100
            motif_length = 10
            trainds, testds = artificial_data(motif_length, domain_pairs,
                                              motif_chance, prom_length,
                                              num_sequences, device)
            net = DeePromoter(para_ker=[27, 14, 7], total_domains=domain_pairs,
                              input_shape=(64, prom_length, 5))
            print(f"#######TRAINING########\ndomain pairs: {domain_pairs}\t"
                  f"motif_chance: {motif_chance}\tnum_sequences: {num_sequences}\t"
                  f"prom_length: {prom_length}\tmotif_length: {motif_length}\n")
            all_accuracies.append(train_function(net, trainds, testds, 1e-2,
                                                 epochs, device, 64,
                                                 DeePromoter=True, verbose=True))

        for num_sequences in [10000, 50000, 90000]:
            motif_chance = 0.3
            domain_pairs = 2
            prom_length = 100
            motif_length = 10
            trainds, testds = artificial_data(motif_length, domain_pairs,
                                              motif_chance, prom_length,
                                              num_sequences, device)
            net = DeePromoter(para_ker=[27, 14, 7], total_domains=domain_pairs,
                              input_shape=(64, prom_length, 5))
            print(f"#######TRAINING########\ndomain pairs: {domain_pairs}\t"
                  f"motif_chance: {motif_chance}\tnum_sequences: {num_sequences}\t"
                  f"prom_length: {prom_length}\tmotif_length: {motif_length}\n")
            all_accuracies.append(train_function(net, trainds, testds, 1e-2,
                                                 epochs, device, 64,
                                                 DeePromoter=True, verbose=True))

        for num_sequences in [10000, 50000, 90000]:
            motif_chance = 0.3
            domain_pairs = 2
            prom_length = 100
            motif_length = 20
            trainds, testds = artificial_data(motif_length, domain_pairs,
                                              motif_chance, prom_length,
                                              num_sequences, device)
            net = DeePromoter(para_ker=[27, 14, 7], total_domains=domain_pairs,
                              input_shape=(64, prom_length, 5))
            print(f"#######TRAINING########\ndomain pairs: {domain_pairs}\t"
                  f"motif_chance: {motif_chance}\tnum_sequences: {num_sequences}\t"
                  f"prom_length: {prom_length}\tmotif_length: {motif_length}\n")
            all_accuracies.append(train_function(net, trainds, testds, 1e-2,
                                                 epochs, device, 64,
                                                 DeePromoter=True, verbose=True))

        for prom_length in [50, 100, 300, 500, 1000]:
            motif_chance = 0.3
            domain_pairs = 2
            num_sequences = 50000
            motif_length = 10
            trainds, testds = artificial_data(motif_length, domain_pairs,
                                              motif_chance, prom_length,
                                              num_sequences, device)
            net = DeePromoter(para_ker=[27, 14, 7], total_domains=domain_pairs,
                              input_shape=(64, prom_length, 5))
            print(f"#######TRAINING########\ndomain pairs: {domain_pairs}\t"
                  f"motif_chance: {motif_chance}\tnum_sequences: {num_sequences}\t"
                  f"prom_length: {prom_length}\tmotif_length: {motif_length}\n")
            all_accuracies.append(train_function(net, trainds, testds, 1e-2,
                                                 epochs, device, 64,
                                                 DeePromoter=True, verbose=True))
        for motif_length in [5, 10, 15, 20, 25]:
            motif_chance = 0.3
            domain_pairs = 2
            num_sequences = 50000
            prom_length = 100

            trainds, testds = artificial_data(motif_length, domain_pairs,
                                              motif_chance, prom_length,
                                              num_sequences, device)
            net = DeePromoter(para_ker=[27, 14, 7], total_domains=domain_pairs,
                              input_shape=(64, prom_length, 5))
            print(f"#######TRAINING########\ndomain pairs: {domain_pairs}\t"
                  f"motif_chance: {motif_chance}\tnum_sequences: {num_sequences}\t"
                  f"prom_length: {prom_length}\tmotif_length: {motif_length}\n")
            all_accuracies.append(train_function(net, trainds, testds, 1e-2,
                                                 epochs, device, 64,
                                                 DeePromoter=True, verbose=True))

        ctoff = 10
        begin_base, end_base = 0, 1250
        print(f'Cutoff: {ctoff}')
        trainds, testds, num_domains = real_data(processed_dir, device,
                                                 begin_base, end_base,
                                                 cutoff=ctoff)
        bal = trainds.get_balance()
        balances = [40 * bal, 45 * bal]
        for balance in balances:
            print(f'Balance: {balance}')
            print(f'Total unique domains in dataset: {num_domains}')
            net = DeePromoter(para_ker=[27, 14, 7], total_domains=num_domains,
                               input_shape=(64, 1251, 5))
            print(f'#########TRAINING#######')
            print(f'EPOCHS: {epochs}, BEGIN BASE: {begin_base}, END BASE: '
                  f'{end_base}, CUTOFF: {ctoff}, NET: {net}\n, BALANCE: {balance}')
            all_accuracies.append(train_function(net, trainds, testds, 1e-2,
                                                 epochs, device, batch_size,
                                                 DeePromoter=True,
                                                 verbose=True,
                                                 pos_weight=balance))
        plot_accuracy(all_accuracies, epochs, save_as_pdf=True)

    else:
        epochs = 100
        batch_size = 64
        begin_base = 800
        end_base = 1000
        cutoff = 1500
        trainds, testds, num_domains = real_data('/home/klis004/nbk_lustre/processed_data', device, begin_base, end_base, cutoff=cutoff)
        net = DeePromoter(para_ker=[27, 14, 7], total_domains=num_domains,
                          input_shape=(64, 201, 5))
        balance = trainds.get_balance()
        all_accuracies.append(train_function(net, trainds, testds, 1e-2, epochs, device, batch_size, DeePromoter=True, verbose=True, pos_weight=balance))


if __name__ == "__main__":
    """The main function of this module"""
    main(artificial=False)