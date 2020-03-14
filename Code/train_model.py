# This takes a few hours to train. Set TRAIN = False to only do inference using a saved model.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

TRAIN = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

data = pd.read_excel('data_freq.xlsx')
x = data["Freq. (GHz)"].values.reshape(-1, 1)
y = data["mean "].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# inds = x_train.reshape(-1).argsort()
# plt.figure(1)
# plt.plot(x_train.reshape(-1)[inds], y_train.reshape(-1)[inds], label="training", linewidth=3)
# inds = x_test.reshape(-1).argsort()
# plt.plot(x_test.reshape(-1)[inds], y_test.reshape(-1)[inds], label="testing", linestyle="dashed", c='r', linewidth=1.5)
# plt.xlabel(r"f (GHz)",fontsize=16)
# plt.ylabel(r"Mean of Maximum Field",fontsize=16)
# plt.tick_params(labelsize=14)
# plt.legend()
# plt.grid()
# plt.savefig('train_test_split.eps', format='eps', dpi=1000)
# plt.show()

class Net_4(nn.Module):
    def __init__(self):
        super(Net_4, self).__init__()
        self.f1 = nn.Linear(1, 100)
        self.f2 = nn.Linear(100, 200)
        self.f3 = nn.Linear(200, 400)
        self.f4 = nn.Linear(400, 200)
        self.out = nn.Linear(200, 1)

    def forward(self, p):
        output = self.f2(torch.tanh(self.f1(p)))
        output = torch.tanh(self.f4(torch.tanh(self.f3(output))))
        return self.out(output)


def compare_with_real(x, y, n_layers="?", n_parameters="?", train=True, fig=1):

    net.eval()
    with torch.no_grad():
        output = net(torch.Tensor(x))
        mse = mean_squared_error(output.numpy(), y)
        plt.figure(fig)
        plt.title("MLP with {} layers and {} parameters: mse = {}".format  (n_layers, n_parameters, round(mse, 5)))
        plt.xlabel(r"f (GHz)",fontsize=16)
        plt.ylabel(r"Mean of Maximum Field",fontsize=16)
        plt.tick_params(labelsize=14)
        inds = x.reshape(-1).argsort()
        if train:
            plt.plot(x.reshape(-1)[inds], y.reshape(-1)[inds], label="Real train function", linewidth=2)
        else:
            plt.plot(x.reshape(-1)[inds], y.reshape(-1)[inds], label="Real test function", linewidth=2)
        plt.plot(x.reshape(-1)[inds], output.numpy().reshape(-1)[inds], label="MLP function", linestyle="dashed", c='r', linewidth=2)
        plt.legend()
        plt.grid()
        plt.savefig('mlp_{}layers_{}.eps'.format(n_layers, "train" if train else "test"), format='eps', dpi=1000)
        plt.show()


if TRAIN:

    x_train_t, x_test_t = torch.Tensor(x_train).to(device), torch.Tensor(x_test).to(device)
    y_train_t, y_test_t = torch.Tensor(y_train).to(device), torch.Tensor(y_test).to(device)
    net = Net_4()
    try:
        net.load_state_dict(torch.load("model5.pt"))
    except:
        print("There was no previous model... starting from scratch!")
    net = net.to(device)
    batch_size = 32
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

    loss_test_best = 0
    for epoch in range(100000):

        net.train()
        loss_train, steps_train = 0, 0
        for k in range(len(x_train_t) // batch_size + 1):
            inds = slice(batch_size * k, batch_size * (k + 1))
            outputs = net(x_train_t[inds])
            loss = criterion(outputs, y_train_t[inds])
            loss_train += loss.item()
            steps_train += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        net.eval()
        loss_test, steps_test = 0, 0
        with torch.no_grad():
            for k in range(len(x_test_t) // batch_size + 1):
                inds = slice(batch_size * k, batch_size * (k + 1))
                outputs = net(x_test_t[inds])
                loss = criterion(outputs, y_test_t[inds])
                loss_test += loss.item()
                steps_test += 1

        if epoch % 100 == 0:
            print("Epoch:", epoch, "| Train loss:", loss_train / steps_train, "- Test loss:", loss_test / steps_test)
            if loss_test_best < loss_test:
                torch.save(net.state_dict(), "model5.pt")
                loss_test = loss_test_best
        if epoch == 1000:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.00005)
        if epoch == 1500:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.000025)
        if epoch == 50000:
            optimizer = torch.optim.Adam(net.parameters(), lr=0.0000005)
    

net = Net_4()
net.load_state_dict(torch.load("model5.pt"))
n_layers = 5
n_parameters = 100 + 100 + 100 * 200 + 200 + 200 * 400 + 400 + 400 * 200 + 200 + 200
compare_with_real(x_train, y_train, n_layers, n_parameters, fig=2)
compare_with_real(x_test, y_test, n_layers, n_parameters, train=False, fig=3)
