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
num_epochs = 20
learning_rate = 0.002
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
VS = pd.read_csv(path + '/input/vs_time.csv')
VS = VS[:data_size]
VS = torch.tensor(VS.values, device = device)

ESED = np.load(os.path.join(path,'input/ES_ED.npy'))
ES_ED = ESED[:data_size]
ES_ED_concatenated = np.concatenate((ES_ED[:, 0], ES_ED[:, 1]), axis=2)


class CardioDataset(torch.utils.data.Dataset):
    def __init__(self, patient_ids, sequences, edvs, esvs, efs, vss, eseds):
        self.patient_ids = patient_ids
        self.sequences = sequences
        self.edvs = edvs
        self.esvs = esvs
        self.efs = efs
        self.vss = vss
        self.eseds = eseds

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        sequence = self.sequences[idx]
        edv = self.edvs[idx]
        esv = self.esvs[idx]
        ef = self.efs[idx]
        esed = self.eseds[idx]
        vs = self.vss[idx]
        
        data = {'patient':patient_id,
                '2CH_sequence': sequence,
                'EDV': edv,
                'ESV': esv,
                'EF': ef,
                'VS': vs,
                'ESED': esed
        }
        return data

train_data = CardioDataset(patient_ids=patient_id, sequences = sequences_all, edvs = LVedv, esvs= LVesv, efs = LVef, vss = VS, eseds=ES_ED_concatenated)

# define normalization layer to make sure output xi in an interval [ai, bi]:
# define normalization layer to make sure output xi in an interval [ai, bi]:
class IntervalNormalizationLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #output = (Tc, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, start_v, start_pla, start_pao). start_v: 20-400; start_pla: 5-20, start_pao: 50-100.
        self.a = torch.tensor([0.5, 0.5, 0.002, 0.01, 0.01, 0.05, 1.0, 3., 0.00001, 20, 5, 50], dtype=torch.float32) #HR in 20-200->Tc in [0.3, 4]
        self.b = torch.tensor([1.5, 1.5, 0.008, 0.10, 0.06, 1.80, 1.6, 6., 0.0015, 400, 20, 100], dtype=torch.float32)
        #taken out (initial conditions): a: 20, 5, 50; b: 400, 20, 100
    def forward(self, inputs):
        sigmoid_output = torch.sigmoid(inputs)
        scaled_output = sigmoid_output * (self.b - self.a) + self.a
        return scaled_output



class ODEFunc(nn.Module): #takes y, t and returns dydt = NN(y,t). NN: defined in self.net (NN: (y**3, t**3):2D->50->tanh->2D=dydt)
  
    def __init__(self, Tc, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls): #cnn: NEW3DCNN
        super(ODEFunc, self).__init__()

        self.Tc = Tc
        self.Rs = Rs
        self.Rm = Rm
        self.Ra = Ra
        self.Rc = Rc
        self.Ca = Ca
        self.Cs = Cs
        self.Cr = Cr
        self.Ls = Ls
        self.Emax = 2.00
        self.Emin = 0.06
    
    def forward(self, t, y):
        dydt = torch.zeros_like(y)

        self.tn =  t / (0.2+0.15*self.Tc) #(t-int(t/self.Tc)*self.Tc) for more loops
        self.P_lv = ((self.Emax-self.Emin)*1.55*(self.tn/0.7)**1.9/((self.tn/0.7)**1.9+1)*1/((self.tn/1.17)**21.9+1) + self.Emin) * y[0]
        dydt[0] = max(y[1]-self.P_lv, 0) / self.Rm - max(self.P_lv-y[3], 0) / self.Ra
        dydt[1] = (y[2]-y[1]) / (self.Rs*self.Cr) - max(y[1]-self.P_lv, 0) / (self.Cr*self.Rm)
        dydt[2] = (y[1]-y[2]) / (self.Rs*self.Cs) + y[4] / self.Cs
        dydt[3] = -y[4]/self.Ca + max(self.P_lv-y[3], 0) / (self.Ca*self.Ra)
        dydt[4] = (y[3] - y[2] - self.Rc * y[4])/self.Ls

        return dydt

import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, num_parameters):
        super(MyModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=256*16*32, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_parameters)
        )
        self.norm1      = IntervalNormalizationLayer()
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = self.norm1(x)
        ini_conds = torch.tensor([x[0][9], x[0][10], x[0][11], x[0][11], torch.tensor(0)])

        func = ODEFunc(x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[0][5], x[0][6], x[0][7], x[0][8]).to(device)
    
        pred_y = odeint(func, ini_conds, t, atol=1e-6, rtol=1e-6).to(device)

        sim_edv = pred_y[int(func.Tc * args.data_size / args.int_time)][0] + 10 #pred_y[n][0][m]: n frame, m-th component (m=0: Vlv-Vd)
        sim_esv = pred_y[int((0.2+0.15*func.Tc)*args.data_size / args.int_time)][0] + 10
        sim_ef = (pred_y[0][0] - pred_y[int((0.2+0.15*func.Tc)*args.data_size / args.int_time)][0]) / pred_y[0][0]
        
        return sim_edv, sim_esv, sim_ef, x


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



model = MyModel(num_parameters=12)
model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#0.97: controls how much weight given to the most recent iterations (larger -> more weight to recent values)
time_meter = RunningAverageMeter(0.97) #measures avg time for running each batch
loss_meter = RunningAverageMeter(0.97) #keep track of avg loss or error during the same iterations

end = time.time()
# Create the data loader

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Training
num_epochs = num_epochs
loss_list = [] 
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for j, batch in enumerate(train_loader):
        optimizer.zero_grad()
        seq = batch["ESED"]
        seq = seq.reshape(batch_size, 1, 256, 512)
        input_tensor = torch.tensor(seq, dtype=torch.float32) 

        #simulated values: sim_output = (V_ED, V_ES)
        loss = 0
        for i in range(batch_size):
           trueEF = torch.tensor([float(batch['EF'][i])])
           trueV_ED = torch.tensor([float(batch['EDV'][i])])
           trueV_ES = torch.tensor([float(batch['ESV'][i])])
           true_SV = torch.tensor([float(batch['VS'][i])])

           input = input_tensor[i].reshape(1, 1, 256, 512)
        
           sim_edv, sim_esv, sim_ef, x = model(input)
           # svloss = torch.square((0.2+0.15*x[0][0] - true_SV)/true_SV*100)
           svloss = 0
           loss += criterion(sim_edv, trueV_ED) + criterion(sim_esv, trueV_ES) + criterion(sim_ef*100, trueEF) + svloss

        # Compute the gradients and update the model parameters using backpropagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() 

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        #loss4: volumes over time (compared to output of segmentation model)

        # Print progress
    loss_list.append(epoch_loss / len(train_loader))  # calculate average epoch loss and append to loss list

    # Print the average epoch loss
    print(f"Epoch {epoch+1}/{num_epochs}: Average Loss = {loss_list[-1]:.4f}")

    if epoch % 1 == 0:
      y0 = torch.tensor([x[0][9].detach(), x[0][10].detach(), x[0][11].detach(), x[0][11].detach(), torch.tensor(0)])
      func = ODEFunc(x[0][0].detach(), x[0][1].detach(), x[0][2].detach(), x[0][3].detach(), x[0][4].detach(), x[0][5].detach(), x[0][6].detach(), x[0][7].detach(), x[0][8].detach()).to(device) #func has a NN: 2->50->tanh->2, and forward(t, y): return NN(y**3)

      pred_y = odeint(func, y0, t, atol=1e-6, rtol=1e-6) #here solve the function from the true_y0 and during the whole time t (and not from random points and short durations batch_time as before)
      vi = pred_y[:, 0].squeeze().tolist()
      ti = t.squeeze().tolist()
      plt.plot(ti, vi, 'b--')

          # Add axis labels and title
      plt.xlabel('t')
      plt.ylabel('Vlv-Vd')
      plt.axvline(x=func.Tc.detach().numpy(), color='r', linestyle='--') #t_ed
      plt.axvline(x=0.2+0.15*func.Tc.detach().numpy(), color='g', linestyle='--') #t_es

          # Show the plot
      plt.show()
            
test_data = next(iter(train_loader))
test_seq = test_data['ESED']
test_seq = test_seq.reshape(batch_size, 1, 256, 512)
test_tensor = torch.tensor(test_seq, dtype=torch.float32) 

# initialize empty lists for sim_EF and trueEF
sim_EF_list = []
trueEF_list = []
sim_EDV_list = []
trueEDV_list = []
sim_ESV_list = []
trueESV_list = []
model_output_list = []

# loop over the batch and calculate sim_EF and trueEF for each sample
for i in range(batch_size):
  trueEF = float(batch['EF'][i])
  trueEDV = float(batch['EDV'][i])
  trueESV = float(batch['ESV'][i])
  input = test_tensor[i].reshape(1, 1, 256, 512)
  sim_edv, sim_esv, sim_ef, x = model(input)
  sim_ef = sim_ef*100
  sim_EF_list.append(sim_ef.detach().numpy())
  sim_EDV_list.append(sim_edv.detach().numpy())
  sim_ESV_list.append(sim_esv.detach().numpy())
  trueEF_list.append(trueEF)
  trueESV_list.append(trueESV)
  trueEDV_list.append(trueEDV)
  model_output_list.append(x.detach().numpy()[0])

# convert the lists to numpy arrays
sim_EF_array = np.array(sim_EF_list)
trueEF_array = np.array(trueEF_list)
sim_EDV_array = np.array(sim_EDV_list)
trueEDV_array = np.array(trueEDV_list)
sim_ESV_array = np.array(sim_ESV_list)
trueESV_array = np.array(trueESV_list)
model_output_array = np.array(model_output_list)

if not os.path.exists(output_path):
    os.makedirs(output_path)

output_file = os.path.join(output_path, 'test_input_frames20.csv')  # set the output file path

# save the data to the output file
np.savetxt(output_file, np.column_stack((sim_EF_array, trueEF_array, sim_EDV_array, trueEDV_array, sim_ESV_array, trueESV_array, model_output_array)), delimiter=',', header='sim_EF,trueEF,sim_EDV,trueEDV,sim_ESV,trueESV,Tc,Rs,Rm,Ra,Rc,Ca,Cs,Cr,Ls,start_v,start_pla,start_pao')


