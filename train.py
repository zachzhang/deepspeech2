import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable

from utils import *
from model import *


batch_size = 32
epochs = 10
num_batches = 100

cuda = True

df = pd.read_csv('training_manifest.csv',index_col=0)

model = DeepSpeech2()

if cuda:
    model = model.cuda()


for e in range(epochs):

    for i in range(num_batches):

        X, y, y_sizes = get_batch(df, batch_size=batch_size)

        X = Variable(torch.from_numpy(np.expand_dims(X,1)), requires_grad=False ).float()
        y = Variable(torch.Tensor(y), requires_grad=False)
        y_sizes = Variable(torch.Tensor(y_sizes), requires_grad=False)

        if cuda:
            X = X.cuda()

        out = model(X)

        loss = criterion(out, y, y_sizes, target_sizes)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
        # SGD step
        optimizer.step()

        if args.cuda:
            torch.cuda.synchronize()

print(model(Variable(torch.from_numpy(np.expand_dims(X,1))).float()).size())

print(y,y_sizes)
