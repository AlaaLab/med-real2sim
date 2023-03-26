#the function simulator generates the functions V_lv and P_lv during N heart cycles (among with 5 other functions, such as pressure in aorta Pao, etc)

import numpy as np
import pandas as pd
from scipy.integrate import odeint #collection of advanced numerical algorithms to solve initial-value problems of ordinary differential equations.
from scipy import interpolate
from matplotlib import pyplot
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
    t = t-int(t/Tc)*Tc
    tn = t/(0.2+0.15*Tc)
    return (Emax-Emin)*1.55*(tn/0.7)**1.9/((tn/0.7)**1.9+1)*1/((tn/1.17)**21.9+1) + Emin
     
''' simulator: solves the ODE and returns 7D vector [Vlv, Plv, Pla, Pa, Pao, Qt, Q_tot] (functions over time interval t)
input values: 
- N: number of HC to simulate
- Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls: static parameters of the circuit
- Emax, Emin, HR, Vd: for elastance
- start_v, start_pla, start_pao: initial conditions (simulation starts at ED)
'''
def simulator(N, HR, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Vd, start_v, start_pla, start_pao):
    Tc = 60/HR #time of 1 cardiac cycle
    start_pa = start_pao
    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig5 in simaan2008dynamical
    y0 = [start_v, start_pla, start_pa, start_pao, start_qt]

    t = np.linspace(0, Tc*N, int(10000*N)) #spaced numbers over interval (start, stop, number_of_steps)

    #obtain 5D vector solution:
    sol = odeint(heart_ode, y0, t, args = (Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc)) #t: list of values

    result_Vlv = np.array(sol[:, 0]) + Vd
    #get Plv from Vlv-Vd
    result_Plv = np.array([Plv(v, Emax, Emin, x, Tc) for x,v in zip(t,sol[:, 0])])
    result_Pla = np.array(sol[:, 1])
    result_Pa = np.array(sol[:, 2])
    result_Pao = np.array(sol[:, 3])
    result_Qt = np.array(sol[:, 4])
    #total current through LV (can be positive or negative):
    Q_tot = [r(x2-P_lv)/Rm-r(P_lv-x4)/Ra for P_lv,x2,x4 in zip(result_Plv, result_Pla, result_Pao)]

    return (result_Vlv, result_Plv, result_Pla, result_Pa, result_Pao, result_Qt, Q_tot, t) #8 functions

#input values for the simulator:

N = float(3) #number of heart cycles shown in the PV loop
HR = float(75) #heart rate (number of cycles per minute)

#static parameters of circuit:
Rs=float(1.0000)
Rm=float(0.0050)
Ra=float(0.0010)
Rc=float(0.0398)
Ca=float(0.0800)
Cs=float(1.3300)
Cr=float(4.4000)
Ls=float(0.0005)

#parameters for elastance:
Emax = float(2) #from simaan2008dynamical (for normal patient)
Emin = float(0.06) #from simaan2008dynamical (for normal patient)
Vd = float(10) #to choose

#initial values:
start_v = float(150) - Vd #from simaan2008dynamical (and v = Vlv - Vd) -> start_plv = start_v*Elastance(Emax, Emin, 0, Tc)
start_pla = float(start_v*Elastance(Emax, Emin, 0, 60/HR) - 0.2) #in simaan2008dynamical: \sim8.2 when start_Vlv=150. we assume start_pla<start_plv 
start_pao =float(77) #from simaan2008dynamical, assumed start_pa = start_pao (initially no flow there)

#receives vector of functions [Vlv, Plv, Pla, Pa, Pao, Qt, Q_tot] (their values over interval of times t):
pv_model= simulator(N, HR, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Vd, start_v, start_pla, start_pao)

Vlv = pv_model[0]
Plv = pv_model[1]
t = pv_model[7]

#plot PV loop:
pyplot.plot(Vlv, Plv) #plot points (Vlv(t), Plv(t))
pyplot.title("PV loop")
pyplot.show()

pyplot.plot(t, Vlv)
pyplot.title("V_lv")
pyplot.show()

pyplot.plot(t, np.array(Plv))
pyplot.title("P_lv")
pyplot.show()
