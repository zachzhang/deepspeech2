import torch
import torch.nn as nn
import math
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable


def seqWise(module, x):
    t, n = x.size(0), x.size(1)
    x = x.view(t * n, -1)
    x = module(x)
    x = x.view(t, n, -1)
    return x

class SeqGRU(nn.Module):
    def __init__(self,rnn_input_size,rnn_hidden_size):
        super(SeqGRU, self).__init__()

        self.h = rnn_hidden_size
        self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=rnn_hidden_size, bidirectional=True, batch_first=True)
        self.bn = nn.BatchNorm1d(rnn_input_size)

    def forward(self,x):
        h0 = Variable(torch.ones(2, x.size()[0], self.h))
        x = seqWise(self.bn , x)
        x = self.rnn(x,h0)[0]
        x = x[:, :, :self.h] + x[:, :,self.h:]

        return x

class DeepSpeech2(nn.Module):
    def __init__(self):
        super(DeepSpeech2, self).__init__()

        rnn_hidden_size = 128
        self._hidden_size = rnn_hidden_size
        rnn_type = nn.LSTM
        nb_layers = 3
        self._hidden_layers = 3
        self._rnn_type = rnn_type

        sample_rate = 16000
        window_size = 0.02

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size - 21) / 2 + 1)
        rnn_input_size *= 32

        print(rnn_input_size)

        rnns = [( 'rnn0', SeqGRU(rnn_input_size,rnn_hidden_size))]

        for i in range(nb_layers):
            rnns.append(( 'rnn_' + str(i+1) , SeqGRU(rnn_hidden_size,rnn_hidden_size) ) )

        self.rnns = nn.Sequential(OrderedDict(rnns))

        self.output = nn.Conv1d(in_channels= rnn_hidden_size, out_channels= 26, kernel_size=1,bias=False)
        self.bn = nn.BatchNorm1d(rnn_hidden_size)


    def forward(self, x):

        x = self.conv(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).contiguous()  # TxNxH

        x = self.rnns(x)

        x = x.transpose(1, 2).contiguous()

        x = self.bn(x)
        x = self.output(x)

        return x