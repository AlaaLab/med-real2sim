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
num_epochs = 100
learning_rate = 0.0001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sequences_all = []
info_data_all = []
path = '/ML/data'
output_path = '/ML/interpolator'


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
class IntervalNormalizationLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # new_output = [start_v, start_paos, emax, emin, rc, rs, cs]
        # output = (Tc =1.0, Rs, Rm = 0.005, Ra = 0.001, Rc, Ca = 0.08, Cs, Cr = 4.44, Ls = 0.0005, start_v, start_pla, start_pao). start_v: 20-400; start_pla: 5-20, start_pao: 50-100.
        self.a = torch.tensor([40, 50, 1, 0.02, 0.025, 0.2, 0.6], dtype=torch.float32) #HR in 20-200->Tc in [0.3, 4]
        self.b = torch.tensor([400, 99, 4.1, 0.1, 0.08, 1.8, 2], dtype=torch.float32)
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


def heart_ode(y, t, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc):
    x1, x2, x3, x4, x5 = y #here y is a vector of 5 values (not functions), at time t, used for getting (dy/dt)(t)
    P_lv = Plv(x1,Emax,Emin,t,Tc)
    dydt = [r(x2-P_lv)/Rm-r(P_lv-x4)/Ra, (x3-x2)/(Rs*Cr)-r(x2-P_lv)/(Cr*Rm), (x2-x3)/(Rs*Cs)+x5/Cs, -x5/Ca+r(P_lv-x4)/(Ca*Ra), (x4-x3-Rc*x5)/Ls]
    return dydt
    
def r(u):
    if u<0:
        return 0
    else:
        return u

#returns Plv at time t using Elastance(t) and Vlv(t)-Vd=x1
def Plv(x1,Emax,Emin,t, Tc):
    return Elastance(Emax,Emin,t, Tc)*x1

#returns Elastance(t)
def Elastance(Emax,Emin,t, Tc):
    t = t-int(t/Tc)*Tc #can remove this if only want 1st ED (and the 1st ES before)
    tn = t/(0.2+0.15*Tc)
    return (Emax-Emin)*1.55*(tn/0.7)**1.9/((tn/0.7)**1.9+1)*1/((tn/1.17)**21.9+1) + Emin

def simulator(N, Tc, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Vd, start_v, start_pla, start_pao):
    start_pa = start_pao
    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig5 in simaan2008dynamical
    y0 = [start_v, start_pla, start_pa, start_pao, start_qt]

    t = np.linspace(0, Tc*N, int(60000*N)) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle
    #changed to 60000 for having integer positions for Tmax
    #obtain 5D vector solution:
    sol = odeint(heart_ode, y0, t, args = (Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc)) #t: list of values
    
    result_Vlv = np.array(sol[:, 0]) + Vd
    ved = result_Vlv[120000]
    ves = result_Vlv[200*int(60/Tc)+9000 + 2 * 60000]
    ef = (ved-ves)/ved * 100

    return ved, ves, ef
    

def f(start_v, start_pao, Emax, Emin, Rc, Rs, Cs):
  #parameters of circuit (assumed constant here):
  N = float(3) #number of heart cycles shown in the PV loop
  Tc = 1.0
  #static parameters of circuit:
  Rm=float(0.0050)
  Ra=float(0.0010)
  Ca=float(0.0800)
  Cr=float(4.4400) #smaller: stretches volume.
  Ls=float(0.0005) #max that makes sense: 0.0015, min that makes sense: 0.00001 (seen checking the pv loops)
  #parameters for elastance:
  Vd = float(10) #to choose

  start_pla = float(start_v*Elastance(Emax, Emin, 0, Tc)) #in simaan2008dynamical: \sim8.2 when start_Vlv=150. we assume start_pla<start_plv 
  
  return simulator(N, Tc, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Vd, start_v, start_pla, start_pao)

start_vs = np.arange(40., 400., 100.)
start_paos = np.arange(50., 99., 12.5)
emaxs = np.arange(1., 4.1, 1.0)
emins = np.arange(0.02, 0.10, 0.025)
rcs = np.arange(0.025, 0.08, 0.007) #now: 8 points
rss = np.arange(0.2, 1.8, 0.5)
css = np.arange(0.6, 2.0, 0.45)

input_vs, input_paos, input_emaxs, input_emins, input_rcs, input_rss, input_css = np.meshgrid(start_vs, start_paos, emaxs, emins, rcs, rss, css, indexing='ij')
input_vs = input_vs.flatten()
input_paos = input_paos.flatten()
input_emaxs = input_emaxs.flatten()
input_emins = input_emins.flatten()
input_rcs = input_rcs.flatten()
input_rss = input_rss.flatten()
input_css = input_css.flatten()

# Create input dataset by stacking the input parameter arrays
input_data = np.vstack([input_vs, input_paos, input_emaxs, input_emins, input_rcs, input_rss, input_css]).T

# Create output dataset by evaluating function for each set of input parameters
output_data = np.zeros((input_data.shape[0],3))
for i in range(input_data.shape[0]):
  ved, ves, ef = f(input_data[i,0], input_data[i,1], input_data[i,2], input_data[i,3], input_data[i,4], input_data[i,5], input_data[i,6])
  output_data[i] = [ved, ves, ef]

print("Input dataset shape: ", input_data.shape)
print("Output dataset shape: ", output_data.shape)
print("done interpolator")

import torch
from torch import nn

# Define the input and output tensors
x = torch.tensor(input_data, dtype = torch.float64) # 7-dimensional input tensor
# x = x.view(64, 3)
y = torch.tensor(output_data, dtype=torch.float64) # 3-dimensional output tensor

# Define a neural network with one hidden layer
class Interpolator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 64).double()
        self.fc2 = nn.Linear(64, 3).double()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the neural network
net = Interpolator()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# Train the neural network
for epoch in range(300000):
    # Forward pass
    y_pred = net(x)
    loss = criterion(y_pred, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, loss: {loss.item():.4f}')


model = NEW3DCNN(num_parameters = 7)
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create the data loader

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Training
num_epochs = num_epochs
for epoch in range(num_epochs):
    for j, batch in enumerate(train_loader):
        optimizer.zero_grad()
        seq = batch["2CH_sequence"]
        seq = seq.reshape(batch_size, 1, 30, 256, 256)
        input_tensor = torch.tensor(seq, dtype=torch.float32) 

        #simulated values: sim_output = (V_ED, V_ES)
        x = model(input_tensor).double()
        output = net(x)
        ved, ves, ef = torch.split(output, split_size_or_sections=1, dim=1)

        trueV_ED = torch.tensor(batch['EDV'])
        trueV_ES = torch.tensor(batch['ESV'])
        true_EF = torch.tensor(batch['EF'])
        criterion = torch.nn.MSELoss()
        loss = criterion(ved,trueV_ED) + criterion(ves, trueV_ES) + criterion(ef, true_EF)

        # Compute the gradients and update the model parameters using backpropagation
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(train_loader), loss.item()/batch_size))




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
test_loader = DataLoader(testing_data, batch_size=50, shuffle=True)

test_data = next(iter(test_loader))
test_seq = test_data['2CH_sequence']
test_seq = test_seq.reshape(50, 1, 30, 256, 256)
test_tensor = torch.tensor(test_seq, dtype=torch.float32) 


x = model(test_tensor).double()
output = net(x)
ved, ves, ef = torch.split(output, split_size_or_sections=1, dim=1)
# initialize empty lists for sim_EF and trueEF
sim_EF = ef
true_EF = test_data['EF']


combined = torch.cat((sim_EF, true_EF.unsqueeze(1)), dim=1)

np_array = combined.detach().numpy()

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_file = os.path.join(output_path, 'test_3NNe100_lr_0.0001_ved.csv')  # set the output file path

# save the data to the output file
np.savetxt(output_file, np_array, delimiter=',', header='sim_EF,trueEF')
