from scipy.integrate._ivp.radau import C
# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import odeint #collection of advanced numerical algorithms to solve initial-value problems of ordinary differential equations.
from scipy import interpolate
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

    N = 6
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

    ved = sol[4 * 60000, 0] + Vd
    ves = sol[200*int(60/Tc)+9000+4*60000, 0] + Vd
    ef = (ved-ves)/ved * 100.

    #ved = Vlv[4 * 60000]
    #ves = Vlv[200*int(60)+9000 + 4 * 60000]
    #ef = (ved-ves)/ved*100

    return ved, ves, ef

n0=6
n1=4
n2=4
n3=8
n4=8
n5=8
n6=1

N = n0*n1*n2*n3*n4*n5*n6
N_test = 50
n_pars = 7
print(N)

# Generate training data
x_train_tc = torch.zeros(N, n_pars)
y_train_tc = torch.zeros(N,3)

tcs = np.linspace(0.5, 2., n0)
startvs = np.linspace(15., 400., n1)
startpaos = np.linspace(5., 150., n2)
rcs = np.linspace(0.05, 4., n3)
emaxs = np.linspace(0.5, 50., n4)
emins = np.linspace(0.02, 0.3, n5)
vds = np.linspace(4., 40., n6)

i = 0
for Tc in tcs:
  for start_v in startvs:
    for start_pao in startpaos:
      for Rc in rcs:
        for Emax in emaxs:
          for Emin in emins:
            for Vd in vds:
              
              x_train_tc[i][0] = Tc
              x_train_tc[i][1] = start_v
              x_train_tc[i][2] = start_pao
              x_train_tc[i][3] = Rc
              x_train_tc[i][4] = Emax
              x_train_tc[i][5] = Emin
              x_train_tc[i][6] = Vd

              ved, ves, ef = f(Tc, start_v, start_pao, Rc, Emax, Emin, Vd)
              y_train_tc[i][0] = ved
              y_train_tc[i][1] = ves
              y_train_tc[i][2] = ef

              i += 1

              '''
              ved(v')=ved(v)-v+v'. ef(v')=(ved(v)-v+v'-ves(v)+v-v')/(ved(v)-v+v')=ef(v)*(ves(v) / ves(v)-v+v'), for a fixed v.
              rel. of v': linear. 
              '''

x_test = np.zeros((N_test, n_pars))
x_test_tc = torch.zeros(N_test, n_pars)
y_test_tc = torch.zeros(N_test,3)

print("Done training data")

for i in range(N_test):
  a = random.uniform(0.5, 2.)
  b = random.uniform(15., 400.)
  c = random.uniform(5., 150.)
  d = random.uniform(0.05, 4.)
  e = random.uniform(0.5, 50.)
  ff = random.uniform(0.02, 0.3)
  g = random.uniform(4., 40.)

  x_test_tc[i][0] = a
  x_test_tc[i][1] = b
  x_test_tc[i][2] = c
  x_test_tc[i][3] = d
  x_test_tc[i][4] = e
  x_test_tc[i][5] = ff
  x_test_tc[i][6] = g

  ved, ves, ef = f(a, b, c, d, e, ff, g)
  y_test_tc[i][0] = ved
  y_test_tc[i][1] = ves
  y_test_tc[i][2] = ef
  
print("Done testing data")

#include Vd in the dataset:
nv = 20
vds = np.linspace(4., 40., nv)

X_train = torch.zeros(N * nv, n_pars)
Y_train = torch.zeros(N * nv, 3)
for i in range(N):
  for j in range(nv):
    for k in range(6):
      X_train[i*nv + j][k] = x_train_tc[i][k] #same first 6 coords
    #for Tc, X_train has nv values for each 6d point:
    X_train[i*nv+j][6] = vds[j]

    #Ved:
    Y_train[i*nv+j][0] = y_train_tc[i][0] - 4. + vds[j]
    #Ves:
    Y_train[i*nv+j][1] = y_train_tc[i][1] - 4. + vds[j]
    #EF:
    Y_train[i*nv+j][2] = y_train_tc[i][2] * y_train_tc[i][0] / (y_train_tc[i][0] - 4. + vds[j]) # = ef(v)*(ved(v) / ved(v)-v+v')

print("Done")

np.save('X_train.npy', X_train)
np.save('Y_train.npy', Y_train)
np.save('x_test_tc.npy', x_test_tc)
np.save('y_test_tc.npy', y_test_tc)

!ls -lh
print("Saved")

#for opening again:
#x_test_tc = np.load('x_test_tc.npy')
