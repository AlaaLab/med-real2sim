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
n2=35

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
emaxs = np.linspace(0.2, 30., n2)
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
  d = random.uniform(0.2, 30.)

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

#once the interpolator NN net is created, run the inverse NN:

#check if can learn from ved, ves the original points:

N = 1000
n_pars = 2

# Generate training data
x_train_tc = torch.zeros(N, n_pars)

for i in range(N):
  a = random.uniform(0.5, 2.)
  b = random.uniform(5, 20)
  c = random.uniform(0.3, 30.)

  ved, ves, ef = f(a, b, 60., 0.08, c, 0.1, 4.)
  x_train_tc[i][0] = ved
  x_train_tc[i][1] = ves

  if (i%50==0): print(i)
  
print("Done training data")

# Define the input and output tensors
x = torch.tensor(x_train_tc, dtype = torch.float64) # 3-dimensional input tensor

# Define a neural network with one hidden layer
class INVNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64).double()
        self.fc2 = nn.Linear(64, 3).double()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the neural network
invnet = INVNN()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(invnet.parameters(), lr=0.01)
losses = []

# Train the neural network
for epoch in range(150000):
    # Forward pass
    x_pred = invnet(x)
    y_pred = net(x_pred)
    loss = criterion(y_pred, x)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, loss: {loss.item():.4f}')
        losses.append(loss.item())

#test and plot real vs predicted ved, ves, and efs:

N_test = 100
x_test_tc = torch.zeros(N_test, 2)
veds_real = []
vess_real = []
efs_real = []
veds_est = []
vess_est = []
efs_est = []

for i in range(N_test):
  a = random.uniform(0.5, 2.)
  b = random.uniform(5., 20.)
  c = 60.
  d = random.uniform(0.3, 30.)
  e = 0.1
  g = 4.

  ved, ves, ef = f(a, b, c, 0.08, d, e, g)
  x_test_tc[i][0] = ved
  x_test_tc[i][1] = ves

  veds_real.append(ved)
  vess_real.append(ves)
  efs_real.append((ved-ves)/ved*100.)

  #y_test_tc[i][2] = ef
  
print("Done testing data")

x = torch.tensor(x_test_tc, dtype = torch.float64) # 7-dimensional input tensor

# Train the neural network
for xi in x:
    # Forward pass
    x_predi = invnet(xi).detach()
    y_predi = net(x_predi)
    loss += abs(y_predi[0].item() - xi[0].item()) + abs(y_predi[1].item() - xi[1].item())

    ved = y_predi[0].item()
    ves = y_predi[1].item()
    
    veds_est.append(ved)
    vess_est.append(ves)
    efs_est.append((ved-ves)/ved*100.)
    
print(loss / (2*N_test))

iters = np.linspace(1, N_test, N_test)
plt.plot(iters, veds_real, color='r')
plt.plot(iters, veds_est, color='b')
plt.show()
plt.plot(iters, vess_real, color='r')
plt.plot(iters, vess_est, color='b')
plt.show()
plt.plot(iters, efs_real, color='r')
plt.plot(iters, efs_est, color='b')
plt.show()
