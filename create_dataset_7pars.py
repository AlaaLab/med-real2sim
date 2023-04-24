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
def f(Tc, start_v, startp, Rc, Emax, Emin, Vd, Ca):

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
    Cs=float(1.3300)
    Cr=float(4.400)
    Ls=float(0.0005)

    sol = odeint(heart_ode, y0, t, args = (Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc)) #t: list of values

    result_Vlv = np.array(sol[:, 0]) + Vd
    result_Plv = np.array([Plv(v, Emax, Emin, xi, Tc) for xi,v in zip(t,sol[:, 0])])

    minv = min(result_Vlv[9*60000:10*60000-1])
    minp = min(result_Plv[9*60000:10*60000-1])
    
    #plt.plot(result_Vlv[9*60000:10*60000], result_Plv[9*60000:10*60000])
    #plt.show()

    ved = sol[9*60000, 0] + Vd
    ves = sol[200*int(60/Tc)+9000+9*60000, 0] + Vd
    ef = (ved-ves)/ved * 100.
    #ved = Vlv[4 * 60000]
    #ves = Vlv[200*int(60)+9000 + 4 * 60000]
    #ef = (ved-ves)/ved*100

    return ved, ves, minv, minp

ts = np.linspace(0.5, 2., 6)
vs = np.linspace(-20., 400., 4)
startps = np.linspace(40., 110., 4)
rcs = np.linspace(0.02, 0.09, 6)
emaxs = np.linspace(0.2, 16., 6)
emins = np.linspace(0.02, 0.1, 6)
vds = np.linspace(4., 15., 4)
cas = np.linspace(0.05, 0.11, 3)

N = 6*4*4*6*6*6*4*3
npars = 8

vedssim = []
vesssim = []
pts = []
errors = [-1]

i = 0
for Vd in vds:
  for Tc in ts:
    for start_v in vs:
      for startp in startps:
        for Rc in rcs:
          for Emax in emaxs:
            for Emin in emins:
              for Ca in cas:
                vedest, vesest, minv, minp = f(Tc, start_v, startp, Rc, Emax, Emin, Vd, Ca)
                vedssim.append(vedest)
                vesssim.append(vesest)
                pts.append([Tc, start_v, startp, Rc, Emax, Emin, Vd, Ca])
                if (minv<=0 or minp<=0): 
                  errors.append(i)
                  print("Error", i)
                if (i%1000==0): print(i)
                i+=1

print("done")
print("errors:", errors)

#convert into torch tensors:
pts2 = torch.zeros(N,npars)
for i in range(N):
  for j in range(npars):
    pts2[i][j] = pts[i][j]

veds2 = torch.zeros(N,2)
for i in range(N):
    veds2[i][0] = vedssim[i]
    veds2[i][1] = vesssim[i]

#save it:
from google.colab import drive
drive.mount('/content/drive')  
output_path = '/content/drive/My Drive/'
file = 'points_8pars'
torch.save(pts2, os.path.join(output_path,f'{file}.pt'))
file = 'veds_8pars'
torch.save(veds2, os.path.join(output_path,f'{file}.pt'))
print("saved")

#create and train the interpolator:
# Define the input and output tensors
x = torch.tensor(pts2, dtype = torch.float64) # 7-dimensional input tensor
# x = x.view(64, 3)
y = torch.tensor(veds2, dtype=torch.float64) # 3-dimensional output tensor

#use it for training the NN:

class Interpolator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 96).double()
        self.fc2 = nn.Linear(96, 2).double()

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = self.fc2(z)
        return z

# Initialize the neural network
net = Interpolator()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
losses = []
d1 = 0
d2 = 0

# Train the neural network
for epoch in range(10000):
    # Forward pass
    y_pred = net(x)
    loss = criterion(y_pred, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if epoch % 5000 == 0:
        print(f'Epoch {epoch}, loss: {loss.item():.4f}')
        losses.append(loss.item())
        
    if (loss.item()<20. and d1==0):
      d1 = 1
      optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
      
    if (loss.item()<4. and d2==0):
      d2 = 1
      optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

print("training error", loss.item())

#for testing the interpolator:
N_test = 3

x_test_tc = torch.zeros(N_test, npars)
y_test_tc = torch.zeros(N_test, 2)

for i in range(N_test):
  x_test_tc[i][0] = random.uniform(0.5, 2.)
  x_test_tc[i][1] = random.uniform(-20., 400.)
  x_test_tc[i][2] = random.uniform(40., 110.)
  x_test_tc[i][3] = random.uniform(0.02, 0.09)
  x_test_tc[i][4] = random.uniform(0.2, 16.)
  x_test_tc[i][5] = random.uniform(0.02, 0.1)
  x_test_tc[i][6] = random.uniform(4., 15.)
  x_test_tc[i][7] = random.uniform(0.05, 0.11)

  #ved, ves, vmin, pmin = f(x_test_tc[i][0], x_test_tc[i][1], x_test_tc[i][2], x_test_tc[i][3], x_test_tc[i][4], x_test_tc[i][5], x_test_tc[i][6], x_test_tc[i][7])
  ved, ves, vmin, pmin = f(x_test_tc[i][0].item(), x_test_tc[i][1].item(), x_test_tc[i][2].item(), x_test_tc[i][3].item(), x_test_tc[i][4].item(), x_test_tc[i][5].item(), x_test_tc[i][6].item(), x_test_tc[i][7].item())
  y_test_tc[i][0] = ved
  y_test_tc[i][1] = ves

error = 0

xt = torch.tensor(x_test_tc, dtype = torch.float64) # 7-dimensional input tensor
# x = x.view(64, 3)
yt = torch.tensor(y_test_tc, dtype=torch.float64) # 3-dimensional output tensor

simveds=[]
simvess=[]
simefs=[]

for i in range(N_test):
  y_pred = net(xt[i])
  vedsim = y_pred[0].detach().item()
  vessim = y_pred[1].detach().item()
  print(vedsim, "real", yt[i][0].item())
  error += abs(vedsim - yt[i][0]) + abs(vessim - yt[i][1])
  simveds.append(vedsim)
  simvess.append(vessim)
  simefs.append((vedsim-vessim)/vedsim*100.)

print("Test error:", error / (N_test*2))

#save the weights of net if it works well enough:
if (error/(N_test*2)<1.):
  import os
  output_path = '/content/drive/My Drive/'
  file = 'model_net_vedves8'
  torch.save(net.state_dict(), os.path.join(output_path,f'{file}__weight.pt'))

#plot the results: (real vs estimated ved, ves, ef)
iters=np.linspace(1,N_test,N_test)
efs=[]
veds=[]
vess=[]
for i in range(N_test):
  veds.append(y_test_tc[i][0])
  vess.append(y_test_tc[i][1])
  efs.append((y_test_tc[i][0]-y_test_tc[i][1])/y_test_tc[i][0]*100)

plt.plot(iters,simveds, color='r')
plt.plot(iters, veds, color='b')
plt.show()
plt.plot(iters,simvess, color='r')
plt.plot(iters, vess, color='b')
plt.show()
plt.plot(iters,simefs, color='r')
plt.plot(iters, efs, color='b')
plt.show()
