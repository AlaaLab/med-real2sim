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

#odesolver:
from torch.storage import T
import argparse
import time

batch_size = 50
data_size = 450
num_epochs = 200
learning_rate = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sequences_all = []
info_data_all = []
path = '/ML/data'
output_path = '/ML/test_output'

# ODE init
parser = argparse.ArgumentParser('ODE demo')

#parameters for the simulation:
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=500) #size of true_y and pred_y and t (n of timepoints for the diff eq in odeint)
#parser.add_argument('--batch_time', type=int, default=100) #batch_t takes the first batch_time(=10) first elements of t
parser.add_argument('--int_time', type=float, default=2.) #added
#parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args(args=[])

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[140., 8.2, 77, 77, 0]]).to(device) #true_y0 = [Vlv-Vd, Pla, Pa, Pao, Qt] at t=0 (in ED)
t = torch.linspace(0., args.int_time, args.data_size).to(device)


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

# define normalization layer to make sure output xi in an interval [ai, bi]:
# define normalization layer to make sure output xi in an interval [ai, bi]:

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

        return x




model = NEW3DCNN(num_parameters = 2)
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Training
num_epochs = num_epochs
loss_list = [] 
for epoch in range(num_epochs):
    epoch_loss = 0.0
    sim_EF_list = []
    true_EF_list = []
    for j, batch in enumerate(train_loader):
        optimizer.zero_grad()
        seq = batch["2CH_sequence"]
        seq = seq.reshape(batch_size, 1, 30, 256, 256)
        input_tensor = torch.tensor(seq, dtype=torch.float32) 

        #simulated values: sim_output = (V_ED, V_ES)
        loss = 0
        for i in range(batch_size):
           trueEF = torch.tensor([float(batch['EF'][i])])
           trueV_ED = torch.tensor([float(batch['EDV'][i])])
           trueV_ES = torch.tensor([float(batch['ESV'][i])])

           input = input_tensor[i].reshape(1, 1, 30, 256, 256)
        
           sim_edv, sim_esv= model(input)[0]
           sim_ef = (sim_edv-sim_esv)/sim_edv
           sim_ef = sim_ef*100
           sim_EF_list.append(sim_ef.detach().numpy())
           true_EF_list.append(trueEF)
           #loss += criterion(sim_edv, trueV_ED) + criterion(sim_esv, trueV_ES) 
           loss += criterion(sim_ef, trueEF)

        # Compute the gradients and update the model parameters using backpropagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() 

        if (epoch+1)%10 == 0 and j == 1:
            sim_EF_array = np.array(sim_EF_list)
            true_EF_array = np.array(true_EF_list)
            print(f"Epoch {epoch+1}/{num_epochs}, Batch 1: Simulated EFs = {np.column_stack((sim_EF_array, true_EF_array))}")
            


        #loss4: volumes over time (compared to output of segmentation model)

        # Print progress
    loss_list.append(epoch_loss / len(train_loader))  # calculate average epoch loss and append to loss list

    # Print the average epoch loss
    print(f"Epoch {epoch+1}/{num_epochs}: Average Loss = {loss_list[-1]:.4f}")
            
test_data = next(iter(train_loader))
test_seq = test_data['2CH_sequence']
test_seq = test_seq.reshape(batch_size, 1, 30, 256, 256)
test_tensor = torch.tensor(test_seq, dtype=torch.float32) 


# initialize empty lists for sim_EF and trueEF
sim_EF_list = []
trueEF_list = []

# loop over the batch and calculate sim_EF and trueEF for each sample
for i in range(batch_size):
  trueEF = float(batch['EF'][i])
  input = input_tensor[i].reshape(1, 1, 30, 256, 256)
  sim_edv, sim_esv = model(input)[0]
  sim_ef = (sim_edv-sim_esv)/sim_edv
  sim_ef = sim_ef*100
  sim_EF_list.append(sim_ef.detach().numpy())
  trueEF_list.append(trueEF)

# convert the lists to numpy arrays
sim_EF_array = np.array(sim_EF_list)
trueEF_array = np.array(trueEF_list)

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_file = os.path.join(output_path, 'test_pv1.csv')  # set the output file path

# save the data to the output file
np.savetxt(output_file, np.column_stack((sim_EF_array, trueEF_array)), delimiter=',', header='sim_EF,trueEF')


