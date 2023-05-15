'''
Model from ferreira2005nonlinear

Parameters from ferreira2005nonlinear vs parameters from simaan2008dynamical
R1 = 1. # = Rs
R2 = 0.005 = Rm
R3 = 0.001 = Ra
R4 = 0.0398 = Rc
C2 = 4.4 = Cr
C3 = 1.33 = Cs
L = 0.0005 = Ls
-> No Ca in this model.
-> One less state variable here (only 4 instead of 5)
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
def heart_ode(y, t, R1, R2, R3, R4, C2, C3, L, Emax, Emin, Tc):
    x1, x2, x3, x4 = y #here y is a vector of 4 values (not functions), at time t, used for getting (dy/dt)(t)
    e_t = Elastance(Emax, Emin, t, Tc)
    dydt = [0, 1/(R1*C2)*(-x2+x3), 1/(R1*C3)*(x2-x3), 0]
    if (t>0.05):
      if (x2 <= x1*e_t and x1*e_t > x3): #ejection
        dydt[0] += -x4
        dydt[2] += x4/C3
        dydt[3] += e_t/L*x1 - x3/L - (R3+R4)*x4/L
      else: #filling
        if (x2 > x1*e_t and x1*e_t < x3):
          dydt[0] += 1/R2*(-x1*e_t+x2)
          dydt[1] += e_t/(R2*C2)*x1 - x2/(R2*C2)
        
    return dydt

#returns Plv at time t using Elastance(t) and Vlv(t)-Vd=x1
def Plv(x1,Emax,Emin,t, Tc):
    return Elastance(Emax,Emin,t, Tc)*x1

#returns Elastance(t)
def Elastance(Emax, Emin, t, Tc):
    t = t-int(t/Tc)*Tc #can remove this if only want 1st ED (and the 1st ES before)
    tn = t/(0.2+0.15*Tc)
    return (Emax-Emin)*1.55*(tn/0.7)**1.9/((tn/0.7)**1.9+1)*1/((tn/1.17)**21.9+1) + Emin
     
''' simulator: solves the ODE and returns 7D vector [Vlv, Plv, Pla, Pa, Pao, Qt, Q_tot] (functions over time interval t)
input values: 
- N: number of HC to simulate
- R1, R2, R3, R4, C2, C3, L: static parameters of the circuit
- Emax, Emin, Tc, Vd: for elastance
- start_v, start_pla, start_pao: initial conditions (simulation starts at ED)
'''
def simulator(N, Tc, R1, R2, R3, R4, C2, C3, L, Emax, Emin, Vd, start_v, start_pa):
    
    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig4b in ferreira2005nonlinear
    start_pla = start_v*Elastance(Emax, Emin, 0, Tc) #start_pla = start_plv at t=0 (see Fig4a in ferreira2005nonlinear)

    y0 = [start_v, start_pla, start_pa, start_qt]

    t = np.linspace(0, Tc*N, int(60000*N)) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle
    #changed to 60000 for having integer positions for Tmax
    #obtain 4D vector solution:
    sol = odeint(heart_ode, y0, t, args = (R1, R2, R3, R4, C2, C3, L, Emax, Emin, Tc)) #t: list of values
    
    result_Vlv = np.array(sol[:, 0]) + Vd
    #get Plv from Vlv-Vd
    result_Plv = np.array([Plv(v, Emax, Emin, x, Tc) for x,v in zip(t,sol[:, 0])])
    result_Pla = np.array(sol[:, 1])
    result_Pa = np.array(sol[:, 2])
    result_Qt = np.array(sol[:, 3])
    
    '''
    plt.plot(t[6*60000:10*60000], result_Pa[6*60000:10*60000], color='red')
    plt.plot(t[6*60000:10*60000], result_Plv[6*60000:10*60000], color='orange')
    plt.plot(t[6*60000:10*60000], result_Pla[6*60000:10*60000], color='green')
    plt.axvline(x=Tc*8, color='r', linestyle='--') #beginning of the second (real) pv loop, corresponds to ED (end of filling phase)
    plt.axvline(x=0.2+9.15*Tc, color='g', linestyle='--') #ES (end of ejection)
    plt.show()
    '''

    return (result_Vlv, result_Plv, result_Pla, result_Pa, result_Qt, t) #8 functions

#input values for the simulator:

N = 12.
HR = 75
Tc = 60/HR

#static parameters of circuit:
R1 = 1. # Rs
R2 = 0.005 # Rm
R3 = 0.001 # Ra
R4 = 0.0398 # Rc
C2 = 4.4 # Cr
C3 = 1.33 # Cs
L = 0.0005 # Ls

#parameters for elastance:
Emax = 2.
Emin = 0.06
Vd = 11.

start_v = 140.
start_pa = 80. #taken from ferreira2005nonlinear (see Fig4a with Pao=Pa at t=0, since x4(t=0)=0)

#receives vector of functions [Vlv, Plv, Pla, Pa, Pao, Qt, Q_tot] (their values over interval of times t):
pv_model= simulator(N, Tc, R1, R2, R3, R4, C2, C3, L, Emax, Emin, Vd, start_v, start_pa)

Vlv = pv_model[0]
P_lv = pv_model[1]
t = pv_model[5]

ved = Vlv[4*60000]
ves = Vlv[200*int(HR)+9000+4*60000]
ef = (ved - ves)/ved*100.
print("Vlv_ED = ", ved)
print("Vlv_ES = ", ves)
print( ef )

#plot PV loop:
pyplot.plot(Vlv, P_lv) #plot points (Vlv(t), Plv(t))
pyplot.xlabel("V_lv (ml)")
pyplot.ylabel("P_lv (mmHg)")
pyplot.show()

pyplot.plot(t, Vlv)
pyplot.ylabel("V_lv(t) (ml)")
pyplot.xlabel("t (s)")
pyplot.axvline(x=Tc, color='r', linestyle='--') #beginning of the second (real) pv loop, corresponds to ED (end of filling phase)
pyplot.axvline(x=0.2+1.15*Tc, color='g', linestyle='--') #ES (end of ejection)
pyplot.xticks([Tc, 0.2+1.15*Tc, 0, 0.5, 1.0, 1.5, 2.0], ['ED', 'ES', '0.0', '0.5', '1.0', '1.5', '2.0'], rotation=0, fontsize=10)
#pyplot.axhline(y=Vlv[60000], color='r', linestyle='--') #ED
#pyplot.axhline(y=Vlv[200*int(HR)+69000], color='b', linestyle='--') #ES. Tmax=Tc+0.2+0.15*Tc=1.15*Tc+0.2-> in frames: (0.2+1.15*Tc)*60000/Tc=0.2*60000/(60/HR)+1.15*60000=200*HR+69000
pyplot.show()

pyplot.plot(t, np.array(P_lv))
pyplot.ylabel("P_lv(t) (mmHg)")
pyplot.xlabel("t (s)")
pyplot.show()
