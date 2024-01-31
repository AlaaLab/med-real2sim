# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=609538&tag=1: [22] in simaan2008dynamical; 'H=P0-Pi'

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
    Ri = 0.0677
    R0 = 0.0677
    Rk = 0
    x1bar = 1.
    alpha = -3.5
    Li = 0.0127
    L0 = 0.0127
    b0 = -0.17070
    b1 = -0.02177
    b2 = 9.9025e-7
    ratew = 6000/60

    x1, x2, x3, x4, x5, x6, x7 = y #here y is a vector of 5 values (not functions), at time t, used for getting (dy/dt)(t)
    
    P_lv = Plv(x1,Emax,Emin,t,Tc)
    if (P_lv <= x1bar): Rk = alpha * (P_lv - x1bar)
    Lstar = Li + L0 + b1
    Lstar2 = -Li -L0 +b1
    Rstar = Ri + + R0 + Rk + b0

    dydt = [-x6 + r(x2-P_lv)/Rm-r(P_lv-x4)/Ra, (x3-x2)/(Rs*Cr)-r(x2-P_lv)/(Cr*Rm), (x2-x3)/(Rs*Cs)+x5/Cs, -x5/Ca+r(P_lv-x4)/(Ca*Ra) + x6/Ca, (x4-x3)/Ls-Rc*x5/Ls, -P_lv / Lstar2 + x4/Lstar2 + (Ri+R0+Rk-b0) / Lstar2 * x6 - b2 / Lstar2 * x7**2, ratew]
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
def f(Tc, start_v, startp, Rc, Emax, Emin, Vd, Ca, Rs, Cs):

    N = 10
    w0 = 12000.
    start_pla = float(start_v*Elastance(Emax, Emin, 0, Tc))
    start_pao = start_pla + startp
    start_pa = start_pao
    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig5 in simaan2008dynamical

    #new:
    start_v = 139.2808870427643 - 10.
    start_pla = 8.067784647536387
    start_pa = 85.62904285092272
    start_pao = 85.80291895439164
    start_qt = 4.410534664180395

    y0 = [start_v, start_pla, start_pa, start_pao, start_qt, 260., w0]

    t = np.linspace(0, 60., 10000000) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle
    #changed to 60000 for having integer positions for Tmax
    #obtain 5D vector solution:
    
    Rm=float(0.0050)
    Ra=float(0.0010)
    Cs=float(1.3300)
    Cr=float(4.4000)
    Ls=float(0.0005)

    sol = odeint(heart_ode, y0, t, args = (Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc)) #t: list of values

    result_Vlv = np.array(sol[:, 0]) + Vd
    result_Plv = np.array([Plv(v, Emax, Emin, xi, Tc) for xi,v in zip(t,sol[:, 0])])

    #minv = min(result_Vlv[9*60000:10*60000-1])
    #minp = min(result_Plv[9*60000:10*60000-1])
    
    plt.plot(result_Vlv, result_Plv)
    plt.show()

    #plot w(t)
    plt.plot(t, np.array(sol[:, 6]))
    plt.show()
    
    #plot x6(t)
    plt.plot(t, np.array(sol[:, 5]))
    plt.show()

    ved = sol[9*60000, 0] + Vd
    ves = sol[200*int(60/Tc)+9000+9*60000, 0] + Vd
    ef = (ved-ves)/ved * 100.
    #ved = Vlv[4 * 60000]
    #ves = Vlv[200*int(60)+9000 + 4 * 60000]
    #ef = (ved-ves)/ved*100
    minv = 0
    minp = 1

    return ved, ves, minv, minp

ts = np.linspace(0.5, 2., 6)
vs = np.linspace(-20., 400., 4)
startps = np.linspace(40., 110., 4)
rcs = np.linspace(0.08, 0.1, 6)
emaxs = np.linspace(0.2, 8., 6)
emins = np.linspace(0.02, 0.1, 6)
vds = np.linspace(4., 15., 4)
cas = np.linspace(0.05, 0.11, 3)
rss = np.linspace(0.5, 2., 5)
