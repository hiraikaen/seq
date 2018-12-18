from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
FUTURE = 1000

#print('torch.ava:', torch.cuda.is_available())
#device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU
dtype = torch.double
#dtype = torch.double if torch.cuda.is_available() else torch.float

param_lstm = 25
N_STEP = 50
class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        # self.lstm1 = nn.LSTMCell(1, 51)
        # self.lstm2 = nn.LSTMCell(51, 51)
        # self.linear = nn.Linear(51, 1)
        self.lstm1 = nn.LSTMCell(1, param_lstm)
        self.lstm2 = nn.LSTMCell(param_lstm, param_lstm)
        self.linear = nn.Linear(param_lstm, 1)

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), param_lstm, dtype=dtype, device=device)
        c_t = torch.zeros(input.size(0), param_lstm, dtype=dtype, device=device)
        c_t = torch.zeros(input.size(0), param_lstm, dtype=dtype, device=device)
        h_t2 = torch.zeros(input.size(0), param_lstm, dtype=dtype, device=device)
        h_t2 = torch.zeros(input.size(0), param_lstm, dtype=dtype, device=device)
        c_t2 = torch.zeros(input.size(0), param_lstm, dtype=dtype, device=device)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata_bit_simple.pt')
    input = torch.from_numpy(data[3:, :-1]).to(device)
    target = torch.from_numpy(data[3:, 1:]).to(device)
    test_input = torch.from_numpy(data[:3, :-1]).to(device)
    test_target = torch.from_numpy(data[:3, 1:]).to(device)
    # build the model
    seq = Sequence()
    seq.to(device)
    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    #begin to train
    for i in range(N_STEP):
        print('STEP: ', i)
        def closure():
            time_start = time.time()
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            loss.backward()
            time_elapsed = time.time() - time_start
            print('loss:', loss.item(), 't:', time_elapsed)
            if loss > 100.0:
                print('explosion - loop terminated')
                exit(1)
            return loss
        optimizer.step(closure)
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = FUTURE
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().cpu().numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()
