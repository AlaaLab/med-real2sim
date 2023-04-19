import matplotlib.pylab as plt
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import os
from skimage.transform import rescale, resize
import torch.nn.functional as F
from torch.utils.data import Subset

import time
from scipy.integrate import odeint #collection of advanced numerical algorithms to solve initial-value problems of ordinary differential equations.
from matplotlib import pyplot as plt
import random
import sys

### ODE: for each t (here fixed), gives dy/dt as a function of y(t) at that t, so can be used for integrating the vector y over time
#it is run for each t going from 0 to tmax
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
     
# Define your function here (for example, a 2-variable function)
def f(Tc, start_v, startp, Rc, Emax, Emin, Vd):

    N = 10
    start_pla = float(start_v*Elastance(Emax, Emin, 0, Tc))
    start_pao = start_pla + startp
    start_pa = start_pao
    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig5 in simaan2008dynamical
    y0 = [start_v, start_pla, start_pa, start_pao, start_qt]

    t = np.linspace(0, Tc*N, int(60000*N)) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle
    #changed to 60000 for having integer positions for Tmax
    #obtain 5D vector solution:
    
    Rs=float(1.0000)
    Rm=float(0.0050)
    Ra=float(0.0010)
    Rc=float(0.06)
    Ca=float(0.0800)
    Cs=float(1.3300)
    Cr=float(4.400)
    Ls=float(0.0005)

    sol = odeint(heart_ode, y0, t, args = (Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc)) #t: list of values

    result_Vlv = np.array(sol[:, 0]) + Vd
    result_Plv = np.array([Plv(v, Emax, Emin, xi, Tc) for xi,v in zip(t,sol[:, 0])])
    
    #plt.plot(result_Vlv, result_Plv)
    #plt.show()

    ved = sol[9*60000, 0] + Vd
    ves = sol[200*int(60/Tc)+9000+9*60000, 0] + Vd
    ef = (ved-ves)/ved * 100.
    #ved = Vlv[4 * 60000]
    #ves = Vlv[200*int(60)+9000 + 4 * 60000]
    #ef = (ved-ves)/ved*100

    return ved, ves, ef

#method 1:
n0=20
n1=20
n2=40

#method 2:
'''
n0=8
n1=20
n2=20
'''

N = n0*n1*n2
n_pars = 3
print(N)

# Generate training data
x_train_tc = torch.zeros(N, n_pars)
y_train_tc = torch.zeros(N,2)

tcs = np.linspace(0.5, 2., n0)
startvs = np.linspace(15., 400., n1)
start_pao = 60.
Rc = 0.08
emaxs = np.linspace(0.3, 30., n2)
Emin = 0.1
Vd = 4.

veds=[]
vess=[]
efs=[]

i = 0
for Tc in tcs:
  for start_v in startvs:
    for Emax in emaxs:

            x_train_tc[i][0] = Tc
            x_train_tc[i][1] = start_v
            x_train_tc[i][2] = Emax

            ved, ves, ef = f(Tc, start_v, start_pao, 0.08, Emax, Emin, Vd)
            y_train_tc[i][0] = ved
            y_train_tc[i][1] = ves

            veds.append(ved)
            vess.append(ves)
            efs.append(ef)

            i += 1

            if (i%1000==0): print(i)

            '''
            ved(v')=ved(v)-v+v'. ef(v')=(ved(v)-v+v'-ves(v)+v-v')/(ved(v)-v+v')=ef(v)*(ves(v) / ves(v)-v+v'), for a fixed v.
            rel. of v': linear. 
            '''

iters = np.linspace(1, len(veds), len(veds))

plt.plot(iters, veds, color='r')
plt.plot(iters, vess, color='b')
plt.plot(iters, efs, color='g')
plt.show()

print("Done training data")

# Define the input and output tensors
x = torch.tensor(x_train_tc, dtype = torch.float64) # 7-dimensional input tensor
# x = x.view(64, 3)
y = torch.tensor(y_train_tc, dtype=torch.float64) # 3-dimensional output tensor

# Define a neural network with one hidden layer
class Interpolator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64).double()
        self.fc2 = nn.Linear(64, 2).double()

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = self.fc2(z)
        return z

# Initialize the neural network
net = Interpolator()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
losses = []
d1 = 0
d2 = 0

# Train the neural network
for epoch in range(500000):
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
        losses.append(loss.item())
        
    if (loss.item()<50. and d1==0):
      d1 = 1
      optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
      
    if (loss.item()<8. and d2==0):
      d2 = 1
      optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

#for testing the interpolator:

N_test = 100
n_pars = 3

x_test = np.zeros((N_test, n_pars))
x_test_tc = torch.zeros(N_test, n_pars)
y_test_tc = torch.zeros(N_test, 2)

for i in range(N_test):
  a = random.uniform(0.5, 2.)
  b = random.uniform(15., 400.)
  d = random.uniform(0.3, 30.)

  x_test_tc[i][0] = a
  x_test_tc[i][1] = b
  x_test_tc[i][2] = d

  ved, ves = f(a, b, 60., 0.08, d, 0.1, 4.)
  y_test_tc[i][0] = ved
  y_test_tc[i][1] = ves

error = 0

xt = torch.tensor(x_test_tc, dtype = torch.float64) # 7-dimensional input tensor
# x = x.view(64, 3)
yt = torch.tensor(y_test_tc, dtype=torch.float64) # 3-dimensional output tensor

for i in range(N_test):
  y_pred = net(xt[i])
  print(y_pred[0].item(), "real", yt[i][0].item())
  error += abs(y_pred[0] - yt[i][0]) + abs(y_pred[1] - yt[i][1])

print("Test error:", error / (N_test*2))

#once the interpolator NN net is created, run the 3DCNN:

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
        self.a = torch.tensor([0.5, 15., 0.3], dtype=torch.float32) #HR in 20-200->Tc in [0.3, 4]
        self.b = torch.tensor([2., 400., 30.], dtype=torch.float32)
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

model = NEW3DCNN(num_parameters = 3)
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
        ved, ves = torch.split(output, split_size_or_sections=1, dim=1)
        ef = (ved - ves) / ved * 100.

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
