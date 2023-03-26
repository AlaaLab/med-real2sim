# a neural network to transform ECG data into a latent space state and static features R and L
# the loss function is a squared error loss function between the expected current and true current
import torch
from torch import nn
from torch.utils.data import DataLoader

import utils as utils
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt

data_size = 4000
batch_time = 50
batch_size = 50
niters = 2000
device = 'cuda:0'
n_obs = 100
# device = "cuda" if torch.cuda.is_available() else "cpu"

def simulate_ecg_data(n_patients=50, duration=20, sampling_rate=200, heart_rate_mean=70, heart_rate_std=10):

    heart_rates = np.random.normal(heart_rate_mean, heart_rate_std, n_patients)
    ecg_signals = []

    for heart_rate in heart_rates:
        ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=heart_rate)
        ecg_signals.append(ecg_signal)

    return ecg_signals

def generate_voltage(ecg_data):
    """
    generate a dataframe of voltage conditional on ecg_data using multivariate gaussian distribution

    """
    voltages = []
    for ecg_signal in ecg_data:
        voltage = np.random.normal(2*ecg_signal, 0.2)
        voltages.append(voltage)

    return voltages

def get_signal_rate(ecg_data):

    features = []
    for ecg_signal in ecg_data:
        ecg_processed, info = nk.ecg_process(ecg_signal, sampling_rate=200)
        hrv =  nk.hrv(ecg_processed, sampling_rate=200, show=False)
        features.append(hrv)
    # return the list of hrv metrics
    return features


ecg_data     = simulate_ecg_data(n_patients= n_obs)
ecg_data_np  = np.array(ecg_data).reshape(n_obs, 4000)

voltages     = generate_voltage(ecg_data)
voltages_np  = np.array(voltages).reshape(n_obs, 4000)
voltages_new = voltages_np.T
voltages_true= torch.from_numpy(voltages_new)

signal_rates = get_signal_rate(ecg_data)
hrv          = np.array(signal_rates).reshape(n_obs, 80)
hrv          = pd.DataFrame(hrv)

static_features         = hrv.iloc[:,[0,1]]
static_features.columns = ["R", "L"]

t      = torch.linspace(1.,21.,data_size)

# generate true currents from the true voltages and static features
def generate_currents(static_features, voltages_true, t):
    """
    generate the currents I for each observation
    """
    currents = []
    for i in range(n_obs):
        R = static_features.iloc[i,0]
        L = static_features.iloc[i,1]
        V = voltages_true[:,i]
        I = V / (R * (1 - torch.exp(-t * R / L)))
        currents.append(I)
    return torch.stack(currents, dim=1)
currents_true = generate_currents(static_features, voltages_true, t)

# plot the currents
plt.plot(currents_true[:,1])
plt.plot(currents_true[:,0])
plt.show()

# write the structure of a feedforward neural network that takes in the ecg data 
# and outputs the latent state V and static features
class ECGNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ECGNet, self).__init__()
        # add a normalization layer
        self.norm = nn.BatchNorm1d(input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# define a ECGDataset class to load ECG data and true currents
class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, ecg_data, currents):
        self.ecg_data = ecg_data
        self.currents = currents

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        ecg_data = self.ecg_data[idx]
        currents = self.currents[:,idx]
        return ecg_data, currents


# write a dataloader function to load the ECG data and true currents
ECGData = ECGDataset(ecg_data_np, currents_true)
dataloader = torch.utils.data.DataLoader(ECGData, batch_size=5, shuffle=True)

ECGmodel = ECGNet(input_size=data_size, hidden_size=100, output_size=data_size+2)
optimizer = torch.optim.Adam(ECGmodel.parameters(), lr=0.00001)
L = nn.MSELoss()

# write a function to train the model
def train_model(model, dataloader, optimizer, num_epochs=niters):
    model.train()
    for epoch in range(num_epochs):
        for i, (ecg_data, currents) in enumerate(dataloader):
            ecg_data = ecg_data.float()
            currents = currents.float()
            optimizer.zero_grad()
            outputs = model(ecg_data)
            # get the first 4000 elements of the output as the latent state V
            V = outputs[:,0:4000]
            # get the last 2 elements of the output as the static features
            statics = outputs[:,4000:4002]
            statics = pd.DataFrame(statics.detach().numpy())
            V = V.T
            # generate the predicted currents
            I = generate_currents(statics, V, t, n = batch_size)
            I = I.T
            loss = L(currents[:,1:], I[:,1:])
            loss.backward()
            optimizer.step()
        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, loss.item()))
    return(statics, V, I)



data = train_model(ECGmodel, dataloader, optimizer, num_epochs=1000)


### Results: compare the true and predicted voltages for the first observation
ECG_0 = ECGData[0]
ECG0 = ECG_0[0]
V0 = voltages_true[:,0]
# get the predicted latent state V and static features of the first observation using trained ECGmodel
outputs = ECGmodel(ECG0) 

V0_pred = ECGmodel(ECG0)