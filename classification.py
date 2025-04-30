# Importing Libraries

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
from torch.optim import Adam
from torchsummary import summary
from sklearn.model_selection import train_test_split
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading the dataset
df = pd.read_csv('./framingham.csv')
df.fillna(999, inplace=True)
x = df.drop('TenYearCHD', axis=1)
y = df.TenYearCHD

x = np.array(x)
y = np.array(y)

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=.70, stratify=y, random_state=3)
x_val,x_test,y_val,y_test = train_test_split(x_test, y_test, train_size=.50, random_state=3)

# Building the dataset Class
class dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# Applying the class on my dataset
x_train = dataset(x_train,y_train)
x_val = dataset(x_val,y_val)
x_test = dataset(x_test,y_test)

# Loading the splits into dataloader

load_train = DataLoader(x_train, shuffle=True, batch_size=8)
load_val = DataLoader(x_val, shuffle=True, batch_size=8)
load_test = DataLoader(x_test, shuffle=True, batch_size=8)

# subclassing pytorch nn
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.input_layer = nn.Linear(x.shape[1], 7)
        self.linear = nn.Linear(7,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.input_layer(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

model = MyModel().to(device)

# training the model

total_train_loss_plot = []
total_train_acc_plot = []
total_val_loss_plot = []
total_val_acc_plot = []

epochs =10
opt = Adam(model.parameters(), lr = 1e-3)
loss = nn.BCELoss()

for epoch in range(epochs):
    total_train_loss = 0
    total_train_acc = 0
    total_val_loss = 0
    total_val_acc = 0
    for data in load_train:
        inputs, label = data
        pred = model(inputs).squeeze(1)
        batch_loss = loss(pred, label)
        total_train_loss += batch_loss.item()
        total_train_acc += (pred.round() == label).sum().item()


      # Back propagation
        batch_loss.backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        for val_data in load_val:
            val_inputs, val_label = val_data
            val_pred = model(val_inputs).squeeze(1)
            val_batch_loss = loss(val_pred, val_label)
            total_val_loss += val_batch_loss.item()
            total_val_acc += (val_pred.round() == val_label).sum().item()
    
    total_train_loss_plot.append(round(total_train_loss / len(load_train.dataset), 3))
    total_train_acc_plot.append(round(total_train_acc / len(load_train.dataset) * 100, 3))
    total_val_loss_plot.append(round(total_val_loss / len(load_val.dataset), 3))
    total_val_acc_plot.append(round(total_val_acc / len(load_val.dataset) * 100, 3))
        
