"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import os
import torch.optim as optim
from dataloader import load_dataset, decode_one_seq
import glob
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Function definitions
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(100, 100, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(100, 100, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)

class Generator(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(128, hidden*seq_len)
        self.block = nn.Sequential(
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
        )
        self.conv1 = nn.Conv1d(hidden, n_chars, 1)
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, self.hidden, self.seq_len) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(self.batch_size*self.seq_len, -1)
        output = F.gumbel_softmax(output, 0.5)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))

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

class WGAN():
    def __init__(self, device, batch_size=64, lr=0.0001, num_epochs=10, seq_len=100, data_dir='/home/klis004/thesis/seq2domain/cnn/motif_1x_randomplek.txt', \
        run_name='test', hidden=512, d_steps = 10):
        self.device = device
        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.seq_len = seq_len
        self.d_steps = d_steps
        self.g_steps = 1
        self.lamda = 10
        self.checkpoint_dir = './checkpoint/' + run_name + '/'
        self.sample_dir = './samples/' + run_name + '/'
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        self.load_data(data_dir)
        self.build_model()

    def build_model(self):
        self.G = Generator(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        self.D = Discriminator(len(self.charmap), self.seq_len, self.batch_size, self.hidden)
        self.G.to(self.device)
        self.D.to(self.device)
        self.G_optimizer = optim.Adam(self.G.parameters(), self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(self.D.parameters(), self.lr, betas=(0.5, 0.9))


    def load_data(self, datadir):
        max_examples = 1e6
        lines, self.charmap, self.inv_charmap = load_dataset(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        )
        self.data = lines

    def save_model(self, epoch):
        torch.save(self.G.state_dict(), self.checkpoint_dir + f'G_weights_{epoch}.pth')
        torch.save(self.D.state_dict(), self.checkpoint_dir + f'D_weights_{epoch}.pth')

    def load_model(self, epoch, directory=''):
        if len(directory) == 0:
            directory = self.checkpoint_dir
        list_G = glob.glob(directory + "G*.pth")
        list_D = glob.glob(directory + "D*.pth")
        if len(list_G) == 0:
            print("No saved models found! Starting from scratch.")
            return 1
        G_file = max(list_G, key=os.path.getctime)
        D_file = max(list_D, key=os.path.getctime)
        epoch_found = int((G_file.split('_')[-1]).split('.')[0])
        print(f'Checkpoint from epoch {epoch_found} found! Using those parameters')
        self.G.load_state_dict(torch.load(G_file))
        self.D.load_state_dict(torch.load(D_file))
        return epoch_found

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1, 1)
        alpha = alpha.view(-1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(self.device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True)[0]
        gradients = gradients.contiguous().view(self.batch_size, -1)
        # gradients = gradients.view(self.batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        return self.lamda * ((gradients_norm - 1) ** 2).mean()

    def disc_train_iteration(self, real_data):
        self.D_optimizer.zero_grad()
        fake_data = self.sample_generator(self.batch_size)
        d_fake_pred = self.D(fake_data)
        d_fake_err = d_fake_pred.mean()
        d_real_pred = self.D(real_data)
        d_real_err = d_real_pred.mean()

        gradient_penalty = self.calc_gradient_penalty(real_data, fake_data)

        d_err = d_fake_err - d_real_err + gradient_penalty
        d_err.backward()
        self.D_optimizer.step()

        return d_fake_err.data, d_real_err.data, gradient_penalty.data

    def sample_generator(self, num_sample):
        z_input = torch.randn(num_sample, 128, requires_grad=True)
        z_input = z_input.to(self.device)
        generated_data = self.G(z_input)
        return generated_data

    def gen_train_iteration(self):
        self.G_optimizer.zero_grad()
        z_input = torch.randn(self.batch_size, 128, requires_grad=True).to(self.device)
        g_fake_data = self.G(z_input)
        dg_fake_pred = self.D(g_fake_data)
        g_err = -torch.mean(dg_fake_pred)
        g_err.backward()
        self.G_optimizer.step()
        return g_err

    def train_model(self, load_dir):
        init_epoch = self.load_model(load_dir)
        total_iterations = 10
        losses_f = open(self.checkpoint_dir + 'losses.txt', 'a+')
        d_fake_losses, d_real_losses, grad_penalties = [], [], []
        G_losses, D_losses, W_dist = [], [], []

        table = np.arange(len(self.charmap)).reshape(-1, 1)
        one_hot = OneHotEncoder()
        one_hot.fit(table)

        counter = 0
        for epoch in range(self.num_epochs):
            n_batches = int(len(self.data) / self.batch_size)
            for idx in range(n_batches):
                _data = np.array([[self.charmap[c] for c in l] for l in self.data[idx*self.batch_size:(idx+1)*self.batch_size]], dtype='int32')
                data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(self.batch_size, -1, len(self.charmap))
                real_data = torch.Tensor(data_one_hot).to(self.device)

                d_fake_err, d_real_err, gradient_penalty = self.disc_train_iteration(real_data)
                d_err = d_fake_err - d_real_err + gradient_penalty

                d_fake_np, d_real_np, gp_np = d_fake_err.cpu().numpy(), d_real_err.cpu().numpy(), gradient_penalty.cpu().numpy()
                grad_penalties.append(gp_np)
                d_real_losses.append(d_real_np)
                d_fake_losses.append(d_fake_np)
                D_losses.append(d_fake_np - d_real_np + gp_np)
                W_dist.append(d_real_np - d_fake_np)

                if counter % self.d_steps == 0:
                    g_err = self.gen_train_iteration()
                    G_losses.append((g_err.data).cpu().numpy())

                if counter % 100 == 99:
                    self.save_model(epoch)
                    self.sample(epoch)

                if counter % 10 == 9:
                    print(counter)
                    print(idx)
                    print(epoch)

                    summary_str = f'Iteration [{epoch}/{total_iterations}] - loss_d: {(d_err.data).cpu().numpy()}, loss_g: {(g_err.data).cpu().numpy()}, w_dist: {((d_real_err - d_fake_err).data).cpu().numpy()}, grad_penalty: {gp_np}\n'
                    print(summary_str)
                    losses_f.write(summary_str)

                    # plot_losses([G_losses, D_losses], ["gen", "disc"],
                    #             self.sample_dir + "losses.png")
                    # plot_losses([W_dist], ["w_dist"],
                    #             self.sample_dir + "dist.png")
                    # plot_losses([grad_penalties], ["grad_penalties"],
                    #             self.sample_dir + "grad.png")
                    # plot_losses([d_fake_losses, d_real_losses],
                    #             ["d_fake", "d_real"],
                    #             self.sample_dir + "d_loss_components.png")

                counter += 1
            np.random.shuffle(self.data)

    def sample(self, epoch):
        z = torch.randn(self.batch_size, 128).to(self.device)
        self.G.eval()
        torch_seqs = self.G(z)
        seqs = (torch_seqs.data).cpu().numpy()
        decoded_seqs = [decode_one_seq(seq, self.inv_charmap) + '\n' for seq in seqs]
        with open(self.sample_dir + f'sampled_{epoch}.txt', 'w+') as f:
            f.writelines(decoded_seqs)
        self.G.train()

def main():
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = WGAN(device=device, run_name='motif_1x_randomplek')
    model.train_model('./new')

if __name__ == "__main__":
    main()
