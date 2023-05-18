'''
Model from stergiopoulos1996determinants
'''

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy import interpolate
from matplotlib import pyplot
import sys
import matplotlib.pyplot as plt

### ODE: for each t (here fixed), gives dy/dt as a function of y(t) at that t, so can be used for integrating the vector y over time
#it is run for each t going from 0 to tmax
def heart_ode(y, t, Pv, Rc, Rp, Rv, C, Emax, Emin, Tc):
    x1, x2 = y #here y is a vector of 4 values (not functions), at time t, used for getting (dy/dt)(t)
    et = Elastance(Emax, Emin, t, Tc)
    dydt = [ relu(Pv - x1*et)/Rv - relu(x1*et - x2)/Rc, relu(x1*et-x2)/(Rc*C) - x2/(Rp*C) ]

    return dydt

def relu(x):
  if (x>0): return x
  else: return 0

#returns Plv at time t using Elastance(t) and Vlv(t)-Vd=x1
def Plv(x1,Emax,Emin,t, Tc):
    return Elastance(Emax,Emin,t, Tc)*x1

#returns Elastance(t)
def Elastance(Emax, Emin, t, Tc):
    t = t-int(t/Tc)*Tc #can remove this if only want 1st ED (and the 1st ES before)
    tn = t/(0.2+0.15*Tc)
    return (Emax-Emin)*1.55*(tn/0.7)**1.9/((tn/0.7)**1.9+1)*1/((tn/1.17)**21.9+1) + Emin
    
def simulator(N, Pv, Rc, Rp, Rv, C, Emax, Emin, Tc, Vd, start_v, start_pa):
    Tc=1.
    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig4b in ferreira2005nonlinear
    start_pa = start_v*Elastance(Emax, Emin, 0, Tc) #start_pla = start_plv at t=0 (see Fig4a in ferreira2005nonlinear)

    y0 = [start_v, start_pa]

    t = np.linspace(0, Tc*N, int(60000*N)) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle
    #changed to 60000 for having integer positions for Tmax
    #obtain 4D vector solution:

    #version 1:
    sol = odeint(heart_ode, y0, t, args = (Pv, Rc, Rp, Rv, C, Emax, Emin, Tc)) #t: list of values
    result_Vlv = np.array(sol[:, 0]) + Vd
    #get Plv from Vlv-Vd
    result_Plv = np.array([Plv(v, Emax, Emin, x, Tc) for x,v in zip(t,sol[:, 0])])
    result_Pa = np.array(sol[:, 1])

    plt.plot(t[6*60000:10*60000], result_Vlv[6*60000:10*60000])
    plt.show()
    plt.plot(result_Vlv[6*60000:10*60000], result_Plv[6*60000:10*60000], color='red')
    plt.show()

    return 0

#input values for the simulator:

N = 20.
HR = 75
Tc = 60/HR

#parameters for elastance:
Emax = 2.31
Emin = 0.06
Vd = 20.

Pv = 7.5
Rc = 0.51
Rp = 1.05
Rv = 0.1
C = 1.60

start_v = 140.
start_pa = 80. #taken from ferreira2005nonlinear (see Fig4a with Pao=Pa at t=0, since x4(t=0)=0)

#receives vector of functions [Vlv, Plv, Pla, Pa, Pao, Qt, Q_tot] (their values over interval of times t):
pv_model= simulator(N, Pv, Rc, Rp, Rv, C, Emax, Emin, Tc, Vd, start_v, start_pa)
