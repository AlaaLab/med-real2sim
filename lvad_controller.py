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

#input:
Tc = 60/75
start_v = 140.
Emax = 1.7

def heart_ode0(y, t, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc):
    x1, x2, x3, x4, x5 = y #here y is a vector of 5 values (not functions), at time t, used for getting (dy/dt)(t)
    P_lv = Plv(x1,Emax,Emin,t,Tc)
    dydt = [r(x2-P_lv)/Rm-r(P_lv-x4)/Ra, (x3-x2)/(Rs*Cr)-r(x2-P_lv)/(Cr*Rm), (x2-x3)/(Rs*Cs)+x5/Cs, -x5/Ca+r(P_lv-x4)/(Ca*Ra), (x4-x3-Rc*x5)/Ls]
    return dydt

def f_nolvad(Tc, start_v, Emax):

    N = 20
    Emin = 0.1
    Vd = 4.
    Rs=float(1.0000)
    Rm=float(0.0050)
    Ra=float(0.0010)
    Rc = 0.08
    Ca=float(0.0800)
    Cs=float(1.3300)
    Cr=float(4.400)
    Ls=float(0.0005)

    start_pla = float(start_v*Elastance(Emax, Emin, 0, Tc))
    start_pao = 60.
    start_pa = start_pao
    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig5 in simaan2008dynamical

    y0 = [start_v, start_pla, start_pa, start_pao, start_qt]

    t = np.linspace(0, Tc*N, int(60000*N)) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle
    #changed to 60000 for having integer positions for Tmax
    #obtain 5D vector solution:
    sol = odeint(heart_ode0, y0, t, args = (Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc)) #t: list of values

    result_Vlv = np.array(sol[:, 0]) + Vd
    result_Plv = np.array([Plv(v, Emax, Emin, xi, Tc) for xi,v in zip(t,sol[:, 0])])

    plt.plot(result_Vlv[18*60000:20*60000], result_Plv[18*60000:20*60000], color='black')

    ved = sol[19*60000, 0] + Vd
    ves = sol[200*int(60/Tc)+9000+19*60000, 0] + Vd
    ef = (ved-ves)/ved * 100.
    minv = min(result_Vlv[19*60000:20*60000-1])
    minp = min(result_Plv[19*60000:20*60000-1])

    return ef

#for the LVAD:
counter = 0
Tc = 1.
npoints = int(Tc * 70 * 10000) #70 heart cycles, and 10000 timepoints for each cycle
#so the end of each cycle happens at the timepoints 10000*n, n whole number
x6vals = np.zeros((npoints))

def getslope(y1, y2, y3, x1, x2, x3):
  sum_x = x1 + x2 + x3
  sum_y = y1 + y2 + y3
  sum_xy = x1*y1 + x2*y2 + x3*y3
  sum_xx = x1*x1 + x2*x2 + x3*x3
  # calculate the coefficients of the least-squares line
  n = 3
  slope = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x)
  return slope

### ODE: for each t (here fixed), gives dy/dt as a function of y(t) at that t, so can be used for integrating the vector y over time
#it is run for each t going from 0 to tmax
def heart_ode(y, t, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc, ratew):
    '''
    #from simaan2008dynamical:
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
    '''
    #b0, b1 rom [22] in simaan2008dynamical:
    Ri = 0.0677
    R0 = 0.0677
    Rk = 0
    x1bar = 1.
    alpha = -3.5
    Li = 0.0127
    L0 = 0.0127
    b0 = -0.296
    b1 = -0.027
    b2 = 9.9025e-7
    '''#made up:
    Ri = 0.0677
    R0 = 0.0677
    Rk = 0
    x1bar = 1.
    alpha = -3.5
    Li = 0.0127
    L0 = 0.0127
    b0 = -0.4
    b1 = -0.03
    b2 = 9.9025e-7
    '''

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
     
def f_lvad(Tc, start_v, Emax, c, slope, w0, x60): #slope is slope0 for w

    N = 70
    Emin = 0.1
    Vd = 4.
    Rs=float(1.0000)
    Rm=float(0.0050)
    Ra=float(0.0010)
    Rc = 0.08
    Ca=float(0.0800)
    Cs=float(1.3300)
    Cr=float(4.400)
    Ls=float(0.0005)

    start_pla = float(start_v*Elastance(Emax, Emin, 0, Tc))
    start_pao = 60.
    start_pa = start_pao
    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig5 in simaan2008dynamical

    y0 = [start_v, start_pla, start_pa, start_pao, start_qt, x60, w0]

    ncycle = 10000
    n = 70 * ncycle
    sol = np.zeros((n, 7))
    t = np.linspace(0., Tc * 70, n)
    for j in range(7):
      sol[0][j] = y0[j]

    result_Vlv = []
    result_Plv = []
    result_x6 = []
    result_x7 = []
    envx6 = []
    timesenvx6 = []

    minx6 = 99999
    tmin = 0
    tlastupdate = 0
    lastw = w0
    update = 1
    ratew = 0 #6000/60

    #solve the ODE step by step by adding dydt*dt:
    for j in range(0, n-1):
      #update y with dydt * dt
      y = sol[j]
      dydt = heart_ode(y, t[j], Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc, ratew)
      for k in range(7):
        dydt[k] = dydt[k] * (t[j+1] - t[j])
      sol[j+1] = sol[j] + dydt
      
      #update the min of x6 in the current cylce. also keep the time at which the min is obtained (for getting the slope later)
      if (minx6 > sol[j][5]):
        minx6 = sol[j][5]
        tmin = t[j]

      #add minimum of x6 once each cycle ends: (works). then reset minx6 to 99999 for calculating again the minimum
      if (j%ncycle==0 and j>1):
        envx6.append(minx6)
        timesenvx6.append(tmin)
        minx6 = 99999
      
      #update w (if 0.005 s. have passed since the last update):
      if (slope<0):
        update = 0
      if (t[j+1] - tlastupdate > 0.005 and slope>0 and update==1): #abs(slope)>0.0001
        # if there are enough points of envelope: calculate slope:
        if (len(envx6)>=3):
          slope = getslope(envx6[-1], envx6[-2], envx6[-3], timesenvx6[-1], timesenvx6[-2], timesenvx6[-3])
          sol[j+1][6] = lastw + c * slope
        #otherwise: take arbitrary rate (see Fig. 16a in simaan2008dynamical)
        else:
          sol[j+1][6] = lastw + 0.005 * slope
        #save w(k) (see formula (8) simaan2008dynamical) and the last time of update t[j] (will have to wait 0.005 s for next update of w)
        tlastupdate = t[j+1]
        lastw = sol[j+1][6]
    
    #save functions and print MAP, CO:
    map = 0
    Pao = []

    for i in range(n):
      result_Vlv.append(sol[i, 0] + Vd)
      result_Plv.append(Plv(sol[i, 0], Emax, Emin, t[i], Tc))
      result_x6.append(sol[i, 5])
      result_x7.append(sol[i, 6])
      Pao.append(sol[i, 3])
    
    '''
    #plot result_Plv, and aortic pressure:
    plt.title("Plv (red), Pao (blue)")
    plt.plot(t, result_Plv, color='r')
    plt.xlabel('t (s)')
    plt.ylabel(Pressures (mmHg))
    plt.plot(t, Pao, color='b')
    plt.show()
    '''

    #plot pv loops:
    plt.title("PV loops (black: no LVAD; blue: with LVAD)")
    plt.plot(result_Vlv, result_Plv, color='blue')
    plt.xlabel('V_LV (ml)')
    plt.ylabel('P_LV (mmHg)')
    plt.show()

    #plot Vlv(t):
    plt.plot(t[48*ncycle: 52*ncycle], result_Vlv[48*ncycle: 52*ncycle])
    plt.title("Vlv(t) ")
    plt.ylabel('V_LV (ml)')
    plt.xlabel('t (s)')
    plt.axvline(x=t[50 * ncycle], color='r', linestyle='--')
    plt.axvline(x=t[50 * ncycle + int(ncycle * 0.2 /Tc + 0.15 * ncycle)], color='r', linestyle='--')
    plt.show()
    
    #get co and ef:
    ved = result_Vlv[50 * ncycle]
    ves = result_Vlv[50 * ncycle + int(ncycle * 0.2 /Tc + 0.15 * ncycle)]
    ef = (ved-ves)/ved*100
    CO = ((ved - ves) * 60/Tc ) / 1000

    #get MAP:
    for i in range(n - 5*ncycle, n):
      map += sol[i, 2]
    map = map/(5*ncycle)

    #plot w(t)
    plt.title("w(t)")
    plt.ylabel('w(t) (rpm)')
    plt.xlabel('t (s)')
    plt.plot(t, result_x7)
    plt.show()
    
    #plot x6(t)
    plt.title("x_6(t)")
    plt.xlabel('Flow through LVAD (ml/s)')
    plt.ylabel('t (s)')
    plt.plot(t, result_x6)
    plt.show()

    return ef, CO, map

c = 0.061  #(in simaan2008dynamical: 0.67, but too fast -> 0.061 gives better shape)
slope0 = 400.  #from simaan2008dynamical
w0 = 13700. #from simaan2008dynamical
x60 = 122. #from simaan2008dynamical

ef_nolvad = f_nolvad(Tc, start_v, Emax)
new_ef, CO, MAP = f_lvad(Tc, start_v, Emax, c, slope0, w0, x60)

print("EF before LVAD:", ef_nolvad)
print("New EF after LVAD:", new_ef, "New CO:", CO, "New MAP:", MAP)

'''
Values that can be arbitrarily chosen for the LVAD:

- c (rate at which w is updated after about 2.5 seconds)
- slope0 (initial slope of w, when there are still not 3 points for calculating the slope of the envelope of x6)
- w0 (initial value of w)
- x60 (initial flow in the LVAD)

- internal values of the LVAD: R0, Ri, Rk (defined by x1bar, alpha), Li, L0, b0, b1, b2. 
-> Taken from simaan2008dynamical but could also take b0, b1, b2 from choi1997modeling ([22] in simaan2008dynamical)
'''
