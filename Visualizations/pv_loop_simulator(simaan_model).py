'''
Given a set of circuit parameters, the function pvloop_simulator plots the pressure-volume (PV) loops of cardiac cycles, 
the pressure (aortic, etc.), and blood flow, and returns information about the system (e.g. ejection fraction).
Based on the circuit used in: Simaan, Marwan A., et al. "A dynamical state space representation and performance analysis of a feedback-controlled rotary left ventricular assist device." IEEE Transactions on Control Systems Technology 17.1 (2008): 15-28.
'''

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import os
from skimage.transform import rescale, resize
import torch.nn.functional as F
from torch.utils.data import Subset
import itertools
from math import prod

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
    return max(u, 0.)

#returns Plv at time t using Elastance(t) and Vlv(t)-Vd=x1
def Plv(x1, Emax, Emin, t, Tc):
    return Elastance(Emax,Emin,t, Tc)*x1

#returns Elastance(t)
def Elastance(Emax,Emin,t, Tc):
    t = t-int(t/Tc)*Tc #can remove this if only want 1st ED (and the 1st ES before)
    tn = t/(0.2+0.15*Tc)
    return (Emax-Emin)*1.55*(tn/0.7)**1.9/((tn/0.7)**1.9+1)*1/((tn/1.17)**21.9+1) + Emin

def pvloop_simulator(Tc, start_v, startp, Rs, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Rm, Ra, Vd, N, plotloops, plotpressures, plotflow):

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

    if plotpressures:
      result_Pla = np.array(sol[:, 1])
      result_Pa = np.array(sol[:, 2])
      result_Pao = np.array(sol[:, 3])
      plt.plot(t[(N-2)*60000:(N)*60000], result_Plv[(N-2)*60000:N*60000], label='LV P')
      plt.plot(t[(N-2)*60000:(N)*60000], result_Pao[(N-2)*60000:N*60000], label='Aortic P')
      plt.plot(t[(N-2)*60000:(N)*60000], result_Pa[(N-2)*60000:N*60000], label='Arterial P')
      plt.plot(t[(N-2)*60000:(N)*60000], result_Pla[(N-2)*60000:N*60000], label='Left atrial P')
      plt.xlabel("Time (s)")
      plt.ylabel("Pressure (mmHg)")
      plt.legend(loc='upper right', framealpha=1)
      plt.show()

    if plotflow:
      result_Q = np.array(sol[:, 4])
      plt.plot(t[(N-2)*60000:(N)*60000], result_Q[(N-2)*60000:N*60000])
      plt.xlabel("Time (s)")
      plt.ylabel("Blood flow (ml/s)")

    return ved, ves, ef, minv, minp, maxp, isperiodic

## example of use:

#select circuit values:
Tc = 1.
start_v = 150.
startp = 75.
Rs = 1.0
Rc = 0.0398
Ca = 0.08
Cs = 1.33
Cr = 4.400
Ls = 0.0005
Emax = 1.7
Emin = 0.05
Rm = 0.005
Ra = 0.002
Vd = 1.

# N: number of cycles simulated before plotting pvloops and getting information
N = 70

pvloop_simulator(Tc, start_v, startp, Rs, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Rm, Ra, Vd, N, True, True, True)
