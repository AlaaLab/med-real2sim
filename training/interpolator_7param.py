from scipy.integrate._ivp.radau import C
# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import itertools
from math import prod
from scipy.integrate import odeint #collection of advanced numerical algorithms to solve initial-value problems of ordinary differential equations.
from scipy import interpolate
from matplotlib import pyplot as plt
import random
import sys
import os
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('--output_path', type=str, default='', help='Output path for saving files')
    return parser.parse_args()

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
def f(Tc, start_v, Emax, Emin, Rm, Ra, Vd, N, plotloops):
    startp = 75.
    Rs = 1.0
    Rc = 0.0398
    Ca = 0.08
    Cs = 1.33
    Cr = 4.400
    Ls = 0.0005

    start_pla = float(start_v*Elastance(Emax, Emin, 0, Tc))
    start_pao = startp
    start_pa = start_pao
    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig5 in simaan2008dynamical
    y0 = [start_v, start_pla, start_pa, start_pao, start_qt]

    t = np.linspace(0, Tc*N, int(60000*N)) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle
    #changed to 60000 for having integer positions for Tmax

    sol = odeint(heart_ode, y0, t, args = (Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc)) #t: list of values

    result_Vlv = np.array(sol[:, 0]) + Vd
    result_Plv = np.array([Plv(v, Emax, Emin, xi, Tc) for xi,v in zip(t,sol[:, 0])])

    ved = sol[(N-1)*60000, 0] + Vd
    ves = sol[200*int(60/Tc)+9000+(N-1)*60000, 0] + Vd
    ef = (ved-ves)/ved * 100.

    minv = min(result_Vlv[(N-1)*60000:N*60000-1])
    minp = min(result_Plv[(N-1)*60000:N*60000-1])
    maxp = max(result_Plv[(N-1)*60000:N*60000-1])

    ved2 = sol[(N-1)*60000 - 1, 0] + Vd
    isperiodic = 0
    if (abs(ved-ved2) > 5.): isperiodic = 1

    if plotloops:
      plt.plot(result_Vlv[(N-2)*60000:(N)*60000], result_Plv[(N-2)*60000:N*60000])
      plt.xlabel("LV volume (ml)")
      plt.ylabel("LV pressure (mmHg)")
      plt.show()

    return ved, ves, ef, minv, minp, maxp, isperiodic

class Interpolator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_pars, n_neurons).double()
        self.fc2 = nn.Linear(n_neurons, 2).double()

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = self.fc2(z)
        return z

def main():
    args = parse_arguments()
    output_path = args.output_path
    N = 70
    ints = [[0.4, 1.7], [0., 280.], [0.5, 3.5], [0.02, 0.1], [0.005, 0.1], [0.0001, 0.25]] #validity interval for each learnable parameter
    nvals_pars = [4, 3, 5, 4, 4, 4] #number of values taken from each parameter
    vds = np.linspace(4., 25., 15) # Vd volumes taken (not used for the interpolator)

    n_pars = len(ints)
    n_points = prod(nvals_pars)

    pars = []
    for i in range(n_pars):
      pars.append(np.linspace(ints[i][0], ints[i][1], nvals_pars[i]))

    points = list(itertools.product(*pars))

    veds = []
    vess = []

    redp0 = []
    redp1 = []
    greenp0 = []
    greenp1 = []

    i=0

    for point in points:
      ved, ves, ef, minv, minp, maxp, isperiodic = f(*point, vds[0], N, False) ##recommended to plot the pv loops when building the intervals
      veds.append(ved)
      vess.append(ves)

      if not (minp<=0 or minv<=0 or isperiodic==1):
        if (maxp>145. or maxp<80. or minp>14. or minp<2.):
      #print("Error: pressure out of normal ranges", point)
          redp0.append(ved)
          redp1.append(ves)
        else: ##in good ranges:
          greenp0.append(ved)
          greenp1.append(ves)

      if i%100==0: print(i, "/", n_points)
      i+=1

#convert into torch tensors:
    points2 = torch.tensor(points)
    tensor1 = torch.tensor(veds)
    tensor2 = torch.tensor(vess)
    vedves = torch.stack((tensor1, tensor2), dim=1)
    tensor3 = torch.tensor(greenp0)
    tensor4 = torch.tensor(greenp1)
    vedves_green = torch.stack((tensor3, tensor4), dim=1)

#save points and ved, ves:
    file = 'points'
    torch.save(points2, os.path.join(output_path,f'{file}.pt'))
    file = 'vedves'
    torch.save(vedves, os.path.join(output_path,f'{file}.pt'))
    file = 'vedves_green'
    torch.save(vedves_green, os.path.join(output_path,f'{file}.pt'))
    print("Saved")

    file = 'points.pt'
    pts = torch.load(output_path + file)
    print("First and last point:", pts[0], pts[-1]) ##check if all points had been saved
    file = 'vedves.pt'
# Load the data points from the file
    vedves = torch.load(output_path + file)
    x = torch.tensor(pts, dtype = torch.float64)
    y = torch.tensor(vedves, dtype=torch.float64)

    n_pars = len(pts[0])
    n_neurons = 256 ##in general 256 is enough
    lr = 0.01
    threshold_train = 1.
    threshold_test = 2.
    n_epochs = 30000

# Initialize the neural network
    net = Interpolator()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    i = 0
    while True:
      for epoch in range(n_epochs):
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

      if abs(losses[-1] - losses[-2])<5.:
        lr = lr / 10.
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
      if (i>10 and losses>50.):
        n_neurons += 24
        net = Interpolator()
      if (i>30): break
      if (losses[-1]<threshold_train): break
      i += 1

    print("Training error", loss.item())
    file_name = 'interp7param_weight.pt'
    torch.save(net.state_dict(), os.path.join(output_path, file_name))

if __name__ == "__main__":
    main()