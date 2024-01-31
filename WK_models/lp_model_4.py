'''
Model from her2018windkessel
'''

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy import interpolate
from matplotlib import pyplot
import sys
import matplotlib.pyplot as plt

def heart_ode(y, t, Pla, Rmvc, Rmvo, Ravc, Ravo, Cao, Lao, Rao, Cpa, Rpa, Emax, Emin, Tc):
    x1, x2, x3, x4 = y #here y is a vector of 4 values (not functions), at time t, used for getting (dy/dt)(t)
    e_t = Elastance(Emax, Emin, t, Tc)

    if (Pla>=x1*e_t): Rmv = Rmvo #if mitral valve is open
    else: Rmv = Rmvc
    if (x1*e_t>=x2): Rav = Ravo #if aortic valve is open
    else: Rav = Ravc

    d1 = (Pla - x1*e_t)/Rmv - (x1*e_t-x2)/Rav
    d2 = (x1*e_t-x2)/(Rav*Cao) - x4/Cao
    d3 = -x3/(Rpa*Cpa) + x4/Cpa
    d4 = 1/Lao*(-x3-x4*Rao+x2)

    dydt = [d1, d2, d3, d4]

    return dydt

def relu(x):
  if (x>0): return x
  else: return 0

def act(x):
  if (x>0): return 1
  else: return 0

#returns Plv at time t using Elastance(t) and Vlv(t)-Vd=x1
def Plv(x1,Emax,Emin,t, Tc):
    return Elastance(Emax,Emin,t, Tc)*x1

#returns Elastance(t)
def Elastance(Emax, Emin, t, Tc):
    t = t-int(t/Tc)*Tc #can remove this if only want 1st ED (and the 1st ES before)
    tn = t/(0.2+0.15*Tc)
    return (Emax-Emin)*1.55*(tn/0.7)**1.9/((tn/0.7)**1.9+1)*1/((tn/1.17)**21.9+1) + Emin
    
def simulator(N, Pla, Rmvc, Rmvo, Ravc, Ravo, Cao, Lao, Rao, Cpa, Rpa, Emax, Emin, Tc, Vd, start_v, start_p):

    start_q = 0
    Pla = start_v*Elastance(Emax, Emin, 0, Tc) #start_pla = start_plv at t=0 (see Fig4a in ferreira2005nonlinear)

    y0 = [start_v, start_p, start_p, start_q]

    t = np.linspace(0, Tc*N, int(60000*N)) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle
    #changed to 60000 for having integer positions for Tmax
    #obtain 4D vector solution:

    sol = odeint(heart_ode, y0, t, args = (Pla, Rmvc, Rmvo, Ravc, Ravo, Cao, Lao, Rao, Cpa, Rpa, Emax, Emin, Tc)) #t: list of values
    result_Vlv = np.array(sol[:, 0]) + Vd
    #get Plv from Vlv-Vd
    result_Plv = np.array([Plv(v, Emax, Emin, x, Tc) for x,v in zip(t,sol[:, 0])])
    plt.plot(result_Vlv[(N-10)*60000:N*60000], result_Plv[(N-10)*60000:N*60000], color='blue')
    plt.show()
    result_Pla = np.array(sol[:, 1])
    result_Pa = np.array(sol[:, 2])
    result_Qt = np.array(sol[:, 3])

    plt.plot(t, result_Qt)
    plt.axhline(y=0.)
    plt.show()
    plt.plot(t, result_Vlv)
    plt.show()

    print("MIN", min(result_Qt[6*60000:10*60000]))

    '''
    plt.plot(t[6*60000:10*60000], result_Pa[6*60000:10*60000], color='red')
    plt.plot(t[6*60000:10*60000], result_Plv[6*60000:10*60000], color='orange')
    plt.plot(t[6*60000:10*60000], result_Pla[6*60000:10*60000], color='green')
    plt.axvline(x=Tc*8, color='r', linestyle='--') #beginning of the second (real) pv loop, corresponds to ED (end of filling phase)
    plt.axvline(x=0.2+9.15*Tc, color='g', linestyle='--') #ES (end of ejection)
    plt.show()
    '''

    return 0

#input values for the simulator:
N = 20
HR = 75
Tc = 60/HR

#static parameters of circuit:
Pla = 10.
Rmvo = 0.01
Rmvc = 1000.
Ravo = 0.002
Ravc = 1000.
Rao = 0.08
Cao = 0.15
Lao = 0.0015
Cpa = 1.5
Rpa = 1.35

#parameters for elastance:
Emax = 1.52
Emin = 0.08
Vd = 10.

start_v = 140.
start_p = 80. #taken from ferreira2005nonlinear (see Fig4a with Pao=Pa at t=0, since x4(t=0)=0)

#receives vector of functions [Vlv, Plv, Pla, Pa, Pao, Qt, Q_tot] (their values over interval of times t):
pv_model= simulator(N, Pla, Rmvc, Rmvo, Ravc, Ravo, Cao, Lao, Rao, Cpa, Rpa, Emax, Emin, Tc, Vd, start_v, start_p)
