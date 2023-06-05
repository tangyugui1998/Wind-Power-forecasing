import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import math
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from scipy.io import loadmat
from model import D2SA_ARNet
import joblib

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)-1):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix,:]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


Ws = loadmat(r'data/WS.mat')
Ws = Ws['ave1'][2:]
I = loadmat(r'data/I.mat')
I = I['ave1'][2:]
S = loadmat(r'data/S.mat')
S = S['ave1'][2:]
Power = loadmat(r'data/WT.mat')
Power = Power['ave1'][1:-1]
Data1 = np.hstack((Power, Ws, I, S))
scaler = MinMaxScaler(feature_range=(0,1))
dataset1 = scaler.fit_transform(Data1)

step = 24
X, y = split_sequence(dataset1, step)

train_X, train_y = X[:-720, :], y[:-720, :]
val_X, val_y = X[-720:-480, :], y[-720:-480, :]
test_X, test_y = X[-480:, :], y[-480:, :]

trainingData = TensorDataset(torch.from_numpy(train_X),
                             torch.from_numpy(train_y))
validData = TensorDataset(torch.from_numpy(val_X),
                          torch.from_numpy(val_y))
BATCH_SIZE = 64
train_dataloader = DataLoader(trainingData, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(validData, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print('Data loading finished')

precision = 1e-8
Model = D2SA_ARNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(Model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, verbose=True, eps=precision)

epoch_num = 100
patience_increase = 20
best_loss = float('inf')

for epoch in range(epoch_num):
    print('epoch: {}/{}'.format(epoch+1, epoch_num))
    print('-' * 20)

    running_loss = 0.0
    Model.train()

    for i, (data, label) in enumerate(train_dataloader):

        pred_label = Model(data)
        pred_label = torch.squeeze(pred_label)
        loss = criterion(pred_label, label)
        running_loss += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
    Loss = running_loss / (i+1)
    print('Train Loss: {:.4f}'.format(Loss))
    print('Train RMSE Loss: {:.4f}'.format(math.sqrt(Loss)))

    running_loss = 0.0
    Model.eval()

    for i, (data, label) in enumerate(valid_dataloader):

        pred_label = Model(data)
        pred_label = torch.squeeze(pred_label)
        loss = criterion(pred_label, label)
        running_loss += loss.item()

    Loss = running_loss / (i+1)
    print('Valid Loss:{:.4f}'.format(Loss))
    print('Valid RMSE Loss:{:.4f}'.format(math.sqrt(Loss)))

    scheduler.step(Loss)

    if Loss + precision < best_loss:
        print('New best validation loss: {:.4f}'.format(Loss))
        best_loss = Loss
        best_weight = copy.deepcopy(Model.state_dict())
        best_epoch = epoch + 1
        patience = patience_increase + epoch
        print('So far patience: ', patience)

torch.save(best_weight, r'saved_models\Model.pkl')
print('Training finish')


print('-'*50)
print('Testing start')
TestingModel = D2SA_ARNet()
TestingModel.load_state_dict(torch.load(r'saved_models\Model.pkl'))
TestingModel.eval()

input_tensor = torch.from_numpy(test_X)
pred_y = TestingModel(input_tensor)
pred_y = torch.squeeze(pred_y)
pred_y = pred_y.detach().numpy()


prediction = scaler.inverse_transform(pred_y)
true = scaler.inverse_transform(test_y)

# metrics
print("mean_square_error:", mean_squared_error(true[:,0], prediction[:,0]))
print("RMSE:", math.sqrt(mean_squared_error(true[:,0], prediction[:,0])))
print("MAE:", mean_absolute_error(true[:,0], prediction[:,0]))
print("R_2", r2_score(true[:,0], prediction[:,0]))


plt.plot(prediction[:,0], label='prediction', color='g')
plt.plot(true[:,0], linestyle='dashed', label='power', color='b')
plt.legend()
plt.show()

