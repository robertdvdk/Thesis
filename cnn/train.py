"""
Trains neural network on artificial and real promoter-domain data

Author: Robert van der Klis

Usage: python3 train.py
"""
# Import statements
from load_data import artificial_data, real_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from DeePromoter import DeePromoter

# Function definitions
def init_weights(m):
    """Function to initialize weights of a neural network linear layer

    Returns:
        None
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def train_function(net, dataset_train, dataset_test, lr, epochs, device,
                   batch_size, DeePromoter=False, verbose=True):
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
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
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
    with PdfPages('multipage_plot.pdf') as pdf:
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

if __name__ == "__main__":
    """The main function of this module"""
    all_accuracies = []
    processed_dir = '/home/klis004/nbk_lustre/processed_data'
    prom_length = 1251
    epochs = 50
    batch_size = 64
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    for ctoff in [10, 50, 100, 500]:
        print(f'Cutoff: {ctoff}')
        trainds, testds, num_domains = real_data(processed_dir, prom_length,
                                                 device, cutoff=ctoff)
        print(f'Total unique domains in dataset: {num_domains}')
        net = DeePromoter(total_domains=num_domains, para_ker=[17, 11, 5],
                          input_shape=(batch_size, prom_length, 5))
        all_accuracies.append(train_function(net, trainds, testds, 1e-2, epochs,
                                             device, batch_size, DeePromoter=True,
                                             verbose=True))
    plot_accuracy(all_accuracies, epochs, save_as_pdf=True)
