import matplotlib.pylab as plt
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import os
from skimage.transform import rescale, resize
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
from scipy.interpolate import interp1d

#for pvloop simulator:
import pandas as pd
from scipy.integrate import odeint 
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot
import sys
import numpy as np

#odesolver:
from torch.storage import T
import argparse
import time

batch_size = 50
data_size = 450
num_epochs = 500
learning_rate = 0.001
ID = '202_CAMUS_7param_Vloss'

file = f"{ID}_epoch_{num_epochs}_lr_{learning_rate}"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sequences_all = []
info_data_all = []
path = '/accounts/biost/grad/keying_kuang/ML/data'
output_path = '/accounts/biost/grad/keying_kuang/ML/interpolator6'


for i in range(1, 6):
    sequences_batch = np.load(f'{path}/sequences_{i}.npy', allow_pickle=True)
    info_data_batch = pd.read_csv(f'{path}/info/info_data_{i}.csv')
    
    sequences_all.append(sequences_batch)
    info_data_all.append(info_data_batch)
    
# Concatenate the sequence arrays into a single array
sequences_all = np.concatenate(sequences_all, axis=0)

# Concatenate the info_data DataFrames into a single DataFrame
info_data_all = pd.concat(info_data_all, axis=0, ignore_index=True)

patient_id = torch.tensor(range(data_size), device=device)
LVedv = torch.tensor(info_data_all['LVedv'].values, device=device)
LVesv = torch.tensor(info_data_all['LVesv'].values, device=device)
LVef = torch.tensor(info_data_all['LVef'].values, device=device)
sequences_all = torch.tensor(sequences_all, device=device)

class CardioDataset(torch.utils.data.Dataset):
    def __init__(self, patient_ids, sequences, edvs, esvs, efs):
        self.patient_ids = patient_ids
        self.sequences = sequences
        self.edvs = edvs
        self.esvs = esvs
        self.efs = efs

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        sequence = self.sequences[idx]
        edv = self.edvs[idx]
        esv = self.esvs[idx]
        ef = self.efs[idx]
        
        data = {'patient':patient_id,
                '2CH_sequence': sequence,
                'EDV': edv,
                'ESV': esv,
                'EF': ef   
        }
        return data

train_data = CardioDataset(patient_ids=patient_id, sequences = sequences_all, edvs = LVedv, esvs= LVesv, efs = LVef)

print("Done loading training data!")

# loading validation set
sequences_all = []
info_data_all = []
sequences_batch = np.load(f'{path}/input/sequences_test.npy', allow_pickle=True)
info_data_batch = pd.read_csv(f'{path}/input/info_data_test.csv')

sequences_all.append(sequences_batch)
info_data_all.append(info_data_batch)
    
# Concatenate the sequence arrays into a single array
sequences_all = np.concatenate(sequences_all, axis=0)

# Concatenate the info_data DataFrames into a single DataFrame
info_data_all = pd.concat(info_data_all, axis=0, ignore_index=True)

patient_id = torch.tensor(range(50), device=device)
LVedv = torch.tensor(info_data_all['LVedv'].values, device=device)
LVesv = torch.tensor(info_data_all['LVesv'].values, device=device)
LVef = torch.tensor(info_data_all['LVef'].values, device=device)
sequences_all = torch.tensor(sequences_all, device=device)

testing_data = CardioDataset(patient_ids=patient_id, sequences = sequences_all, edvs = LVedv, esvs= LVesv, efs = LVef)

print('Done loading test data')

full_dataset = torch.utils.data.ConcatDataset([train_data, testing_data])
print('Done concat data!')

class IntervalNormalizationLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # new_output = [Tc, start_p, Emax, Emin, Rm, Ra, Vd]
        self.a = torch.tensor([0.4, 0., 0.5, 0.02, 0.005, 0.0001, 4.], dtype=torch.float32) #HR in 20-200->Tc in [0.3, 4]
        self.b = torch.tensor([1.7, 280., 3.5, 0.1, 0.1, 0.25, 24.], dtype=torch.float32)
        #taken out (initial conditions): a: 20, 5, 50; b: 400, 20, 100
    def forward(self, inputs):
        sigmoid_output = torch.sigmoid(inputs)
        scaled_output = sigmoid_output * (self.b - self.a) + self.a
        return scaled_output
class NEW3DCNN(nn.Module): 
    def __init__(self, num_parameters): 
        super(NEW3DCNN, self).__init__() 
        
        self.conv1      = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(8)
        self.conv2      = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm3d(16)
        self.conv3      = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm3d(32)
        self.conv4      = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm3d(64)
        self.conv5      = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm3d(128)
        self.pool       = nn.AdaptiveAvgPool3d(1)
        self.fc1        = nn.Linear(128, 512)
        self.fc2        = nn.Linear(512, num_parameters)
        self.norm1      = IntervalNormalizationLayer()
        
    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.batchnorm5(self.conv5(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm1(x)

        
        return x


# Define a neural network with one hidden layer
class Interpolator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 250).double()
        self.fc2 = nn.Linear(250, 2).double()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the neural network
net = Interpolator()
net.load_state_dict(torch.load('/accounts/biost/grad/keying_kuang/ML/interpolator6/interp6_7param_weight.pt'))
print("Done loading interpolator!")

model = NEW3DCNN(num_parameters = 7)
model.to(device)
model.load_state_dict(torch.load('/accounts/biost/grad/keying_kuang/ML/interpolator6/202_CAMUS_7param_Vloss_epoch_500_lr_0.001_weight_best_model.pt'))


full_loader = DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=False)
full_data = next(iter(full_loader))

full_seq = full_data['2CH_sequence']
full_seq = full_seq.reshape(len(full_dataset), 1, 30, 256, 256)
full_tensor = torch.tensor(full_seq, dtype=torch.float32) 

full_EF = full_data['EF']

print("Done creating full tensor!")

full_x = model(full_tensor).double()
full_x1 = full_x[:, :6]
full_Vd = full_x[:, -1:]
full_output1 = net(full_x1)
full_output = full_output1 + full_Vd - 4
a, b = torch.split(full_output, split_size_or_sections=1, dim=1)
full_sim_EF = (a - b) / a * 100
full_EF_np = full_EF.numpy()
full_sim_EF_np = full_sim_EF.detach().numpy()

MAE = np.mean(np.abs(full_EF_np - full_sim_EF_np.flatten()))
file_label = f"MAE:{MAE:.2f}"
combined = torch.cat((full_sim_EF, full_EF.unsqueeze(1), full_x), dim=1)
np_array = combined.detach().numpy()
np.savetxt(f'{output_path}/CAMUS_{file}_{file_label}_parameters.csv', np_array, delimiter=',', header='sim_EF,trueEF, Tc, start_p, Emax, Emin, Rm, Ra, Vd')



