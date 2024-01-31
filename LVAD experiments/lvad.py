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
def lvad_ode(y, t, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc, b0, b1, b2, Li, L0, Ri, R0, alpha, x1_lvad, ratew):
    x1, x2, x3, x4, x5, x6, x7 = y #here y is a vector of 5 values (not functions), at time t, used for getting (dy/dt)(t)
    P_lv = Plv(x1,Emax,Emin,t,Tc)
    Lstar = Li + L0 + b1
    Rk = 0
    if (P_lv <= x1_lvad): Rk = alpha * (P_lv - x1_lvad)
    Rstar = Ri + R0 + Rk + b0
    dydt = [r(x2-P_lv)/Rm-r(P_lv-x4)/Ra - x6, (x3-x2)/(Rs*Cr)-r(x2-P_lv)/(Cr*Rm), (x2-x3)/(Rs*Cs)+x5/Cs, -x5/Ca+r(P_lv-x4)/(Ca*Ra) + x6 / Ca, (x4-x3)/Ls -Rc*x5/Ls, 1/Lstar * P_lv - x4/Lstar  - Rstar * x6 / Lstar - b2 / Lstar * x7**2, ratew]
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
def f_lvad(Tc, start_v, startp, Rc, Emax, Emin, Vd, Ca, Rs, b0, b1, b2, Li, L0, Ri, R0, alpha, x1_lvad, w0, ratew):
    
    x60 = 400.
    start_pla = float(start_v*Elastance(Emax, Emin, 0, Tc))
    start_pao = start_pla + startp
    start_pa = start_pao
    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig5 in simaan2008dynamical
    y0 = [start_v, start_pla, start_pa, start_pao, start_qt, x60, w0]

    t = np.linspace(0., 50., 10000) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle

    Rm=float(0.0050)
    Ra=float(0.0010)
    Cs=float(1.3300)
    Cr=float(4.400)
    Ls=float(0.0005)

    sol = odeint(lvad_ode, y0, t, args = (Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc, b0, b1, b2, Li, L0, Ri, R0, alpha, x1_lvad, ratew)) #t: list of values

    result_Vlv = np.array(sol[:, 0]) + Vd
    result_Plv = np.array([Plv(v, Emax, Emin, xi, Tc) for xi,v in zip(t,sol[:, 0])])
    result_x6 = np.array(sol[:, 5])
    result_x7 = np.array(sol[:, 6])

    plt.plot(result_Vlv, result_Plv)
    plt.show()

    plt.plot(t, result_x6)
    plt.show()

    plt.plot(t, result_x7)
    plt.show()

    return 1

#parameters of LVAD: b0, b1, b2, Li, L0, Ri, R0, alpha, x1_lvad, ut ( ut=w(t)^2) ):
Ri = 0.0677
R0 = 0.0677
Li = 0.0127
L0 = 0.0127
b0 = -0.17070
b1 = -0.02177
b2 = 0.00000099025 #9.9025e-7
alpha = -3.5
x1_lvad = 1.
#w0 = 200.
#ratew = 100./60.
w0 = 200
ratew = 100./60.
Tc = 60/75

start_v = 140.
startp = 60.
Rc = 0.0398
Emax = 2.
Emin = 0.06
Vd = 10.
Ca = 0.08
Rs = 1.

f_lvad(Tc, start_v, startp, Rc, Emax, Emin, Vd, Ca, Rs, b0, b1, b2, Li, L0, Ri, R0, alpha, x1_lvad, w0, ratew)
