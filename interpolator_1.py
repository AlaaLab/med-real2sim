from scipy.integrate import odeint #collection of advanced numerical algorithms to solve initial-value problems of ordinary differential equations.
from scipy.interpolate import RegularGridInterpolator
from scipy import interpolate
from matplotlib import pyplot
import sys
import numpy as np

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

def simulator(N, Tc, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Vd, start_v, start_pla, start_pao):
    start_pa = start_pao
    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig5 in simaan2008dynamical
    y0 = [start_v, start_pla, start_pa, start_pao, start_qt]

    t = np.linspace(0, Tc*N, int(60000*N)) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle
    #changed to 60000 for having integer positions for Tmax
    #obtain 5D vector solution:
    sol = odeint(heart_ode, y0, t, args = (Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc)) #t: list of values
    
    result_Vlv = np.array(sol[:, 0]) + Vd
    
    return result_Vlv[120000]

def f(start_v, start_pao, Emax, Emin, Rc, Rs, Cs):
  #parameters of circuit (assumed constant here):
  N = float(3) #number of heart cycles shown in the PV loop
  Tc = 1.0
  #static parameters of circuit:
  Rm=float(0.0050)
  Ra=float(0.0010)
  Ca=float(0.0800)
  Cr=float(4.4400) #smaller: stretches volume.
  Ls=float(0.0005) #max that makes sense: 0.0015, min that makes sense: 0.00001 (seen checking the pv loops)
  #parameters for elastance:
  Vd = float(10) #to choose

  start_pla = float(start_v*Elastance(Emax, Emin, 0, Tc)) #in simaan2008dynamical: \sim8.2 when start_Vlv=150. we assume start_pla<start_plv 
  
  return simulator(N, Tc, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Vd, start_v, start_pla, start_pao)

start_vs = np.arange(40., 400., 100.)
start_paos = np.arange(50., 99., 12.5)
emaxs = np.arange(1., 4.1, 1.0)
emins = np.arange(0.02, 0.10, 0.025)
rcs = np.arange(0.025, 0.08, 0.007) #now: 8 points
rss = np.arange(0.2, 1.8, 0.5)
css = np.arange(0.6, 2.0, 0.45)

data = np.zeros((4,4,4,4,8,4,4))

for i in range(4):
  if (i==3): print("done i")
  for j in range(4):
    if (j==3): print("done j")
    for k in range(4):
      for l in range(4):
        for m in range(8):
          for n in range(4):
            for q in range(4):
              
              data[i][j][k][l][m][n][q] = f(start_vs[i], start_paos[j], emaxs[k], emins[l], rcs[m], rss[n], css[q])

interp = RegularGridInterpolator((start_vs, start_paos, emaxs, emins, rcs, rss, css), data, method='pchip', bounds_error=False, fill_value=None)

print("done")

err = 0
ntot = 500
for i in range(ntot):
  start_v = random.uniform(40., 400.)
  start_pao = random.uniform(50., 100.)
  Emax = random.uniform(1., 4.)
  Emin = random.uniform(0.02, 0.10)
  Rc = random.uniform(0.025, 0.075) #at least check in the interval closer to the interpolated one ([0.025, 0.074])
  Rs = random.uniform(0.2, 1.8)
  Cs = random.uniform(0.6, 2.0)

  ved1 = interp([start_v, start_pao, Emax, Emin, Rc, Rs, Cs])
  v2 = interp([start_v, start_pao, Emax, Emin, Rc + 0.00001, Rs, Cs])

  vedreal = f(start_v, start_pao, Emax, Emin, Rc, Rs, Cs)
  vreal2 = f(start_v, start_pao, Emax, Emin, Rc + 0.00001, Rs , Cs)

  gradsim = (v2 - ved1) / 0.00001
  gradreal = (vreal2 - vedreal) / 0.00001

  err += abs(gradsim - gradreal) / abs(gradreal)

print("Average error in dV_ED/dRc: ", err/ntot)
