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
num_epochs = 15
learning_rate = 0.003
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sequences_all = []
info_data_all = []
path = '/accounts/biost/grad/keying_kuang/ML/data'
output_path = '/accounts/biost/grad/keying_kuang/ML/output2'


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
    
    return result_Vlv[120000]

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

data = np.zeros((4,4,4,4,8,4,4))

for i in range(4):
  if (i==3): print("done i")
  for j in range(4):
    if (j==3): print("done j")
    for k in range(4):
      for l in range(4):
        for m in range(8):
          for n in range(4):
            for q in range(4):
              
              data[i][j][k][l][m][n][q] = f(start_vs[i], start_paos[j], emaxs[k], emins[l], rcs[m], rss[n], css[q])

interp = RegularGridInterpolator((start_vs, start_paos, emaxs, emins, rcs, rss, css), data, method='pchip', bounds_error=False, fill_value=None)

print("done interpolator")



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
        loss = 0
        for i in range(batch_size):
           # trueEF = torch.tensor([float(batch['EF'][i])])
           trueV_ED = torch.tensor([float(batch['EDV'][i])])
           # trueV_ES = torch.tensor([float(batch['ESV'][i])])

           input = input_tensor[i].reshape(1, 1, 30, 256, 256)
           x = model(input)
        
           x_detached = x.detach().numpy().astype(np.float32)
           ved1 = interp([x_detached[0][0], x_detached[0][1], x_detached[0][2], x_detached[0][3], x_detached[0][4], x_detached[0][5], x_detached[0][6]])
           ved1 = np.array(ved1, dtype=np.float64)
           ved = torch.tensor(ved1, dtype=torch.double, requires_grad=True)
           loss += criterion(ved.double(), trueV_ED.double())
        # Compute the gradients and update the model parameters using backpropagation
        loss.backward()
        optimizer.step()

        #loss4: volumes over time (compared to output of segmentation model)

        # Print progress
    if epoch % 1 == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(train_loader), loss.item()/batch_size))
            
test_data = next(iter(train_loader))
test_seq = test_data['2CH_sequence']
test_seq = test_seq.reshape(batch_size, 1, 30, 256, 256)
test_tensor = torch.tensor(test_seq, dtype=torch.float32) 


# initialize empty lists for sim_EF and trueEF
sim_ED_list = []
trueED_list = []

# loop over the batch and calculate sim_EF and trueEF for each sample
for i in range(batch_size):
  trueED = float(batch['EDV'][i])
  input = input_tensor[i].reshape(1, 1, 30, 256, 256)
  x = model(input)
  x_detached = x.detach().numpy().astype(np.float32)
  ved1 = interp([x_detached[0][0], x_detached[0][1], x_detached[0][2], x_detached[0][3], x_detached[0][4], x_detached[0][5], x_detached[0][6]])
  ved1 = np.array(ved1, dtype=np.float64)
  sim_ED_list.append(ved1)
  trueED_list.append(trueED)

# convert the lists to numpy arrays
sim_EF_array = np.array(sim_ED_list)
trueEF_array = np.array(trueED_list)

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_file = os.path.join(output_path, 'model3_15e.csv')  # set the output file path

# save the data to the output file
np.savetxt(output_file, np.column_stack((sim_EF_array, trueEF_array)), delimiter=',', header='sim_ED,trueED')


