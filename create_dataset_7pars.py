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

ts = np.linspace(0.7, 1.3, 6)
vs = np.linspace(-20., 400., 3)
startps = np.linspace(40., 110., 3)
rcs = np.linspace(0.08, 0.09, 5)
emaxs = np.linspace(0.2, 16., 6)
emins = np.linspace(0.02, 0.1, 6)
vds = np.linspace(4., 15., 4)

N = 6*3*3*5*6*6*4

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
              #print(i, ":", Tc, start_v, startp, Rc, Emax, Emin, Vd)
              vedest, vesest, minv, minp = f(Tc, start_v, startp, Rc, Emax, Emin, Vd)
              vedssim.append(vedest)
              vesssim.append(vesest)
              pts.append([Tc, start_v, startp, Rc, Emax, Emin, Vd])
              if (minv<=0 or minp<=0): 
                errors.append(i)
                print("Error", i)
              if (i%1000==0): print(i)
              i+=1

print("done")
              
output_path = '/content/drive/My Drive/'

file = 'points_7pars'
pts2 = torch.zeros(N,7)
for i in range(N):
  for j in range(7):
    pts2[i][j] = pts[i][j]
torch.save(pts2, os.path.join(output_path,f'{file}.pt'))

file = 'veds_7pars'
vedssim2 = torch.zeros(N)
for i in range(NotADirectoryError):
    vedssim2[i] = vedssim[i]
torch.save(vedssim2, os.path.join(output_path,f'{file}.pt'))

file = 'vess_7pars'
vesssim2 = torch.zeros(N)
for i in range(N):
    vesssim2[i] = vesssim[i]
torch.save(vesssim, os.path.join(output_path,f'{file}.pt'))

print("saved")

print("errors:", errors)
