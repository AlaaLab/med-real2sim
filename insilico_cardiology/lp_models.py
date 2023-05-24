

class ferreira2005nonlinear:
  def __init__(self):
    self.n_parameters = 17
    self.name_parameters = ['Tc', 'R1', 'R2', 'R3', 'R4', 'C2', 'C3', 'L', 'Emax', 'Emin', 'n1', 'n2', 'alpha1', 'alpha2', 'Vd', 'start_v', 'start_pa']
    self.parameters = [60/75, 1., 0.005, 0.001, 0.0398, 4.4, 1.33, 0.000005, 2., 0.06, 1.9, 21.9, 0.7, 1.17, 11., 140., 80.]
    self.validity_intervals = [[0.5, 2.], [0.5, 2.5], [0.001, 1.], [0.001, 0.1], [0.03, 0.1], [3.5, 80.], [1., 2.], [0.000005, 0.0001], [0.5, 8.], [0.01, 0.3], [1., 4.], [10., 35.], [0.7, 0.7], [1.17, 1.17], [4., 40.], [-20., 350.], [60., 90.]]
    self.learnable_parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]

  def Elastance(self, t, Tc, Emax, Emin, n1, n2, alpha1, alpha2):
    tn = (t-int(t/Tc)*Tc)/(0.2+0.15*Tc)
    et = (Emax-Emin)*1.55*(tn/(alpha1))**n1/((tn/(alpha1))**n1+1)*1/((tn/(alpha2))**n2+1) + Emin
    
    return et

  def heart_ode0(self, y, t, Tc, R1, R2, R3, R4, C2, C3, L, Emax, Emin): #n1, n2, alpha1, alpha2
    x1, x2, x3, x4 = y #here y is a vector of 4 values (not functions), at time t, used for getting (dy/dt)(t)
   
    tn = t-int(t/Tc)*Tc
    #e_t = (Emax-Emin)*1.55*(tn/(alpha1))**n1/((tn/(alpha1))**n1+1)*1/((tn/(alpha2))**n2+1) + Emin
    e_t = (Emax-Emin)*1.55*(tn/0.7)**1.9/((tn/0.7)**1.9+1)*1/((tn/1.17)**21.9+1) + Emin

    dydt = [0, 1/(R1*C2)*(-x2+x3), 1/(R1*C3)*(x2-x3), 0]
    
    if (t>0.05):
      if (x2 <= x1*e_t and x1*e_t >= x3): #ejection x1*e_t >= x3
        dydt[0] += -x4
        dydt[2] += x4/C3
        dydt[3] += e_t/L*x1 - x3/L - (R3+R4)*x4/L
      else: #filling
        if (x2 >= x1*e_t and x1*e_t <= x3):
          dydt[0] += 1/R2*(-x1*e_t+x2)
          dydt[1] += e_t/(R2*C2)*x1 - x2/(R2*C2)

    return dydt

  def heart_ode(self, y, t, Tc, R1, R2, R3, R4, C2, C3, L, Emax, Emin, n1, n2, alpha1, alpha2):
    x1, x2, x3, x4 = y #here y is a vector of 4 values (not functions), at time t, used for getting (dy/dt)(t)
    t = t-int(t/Tc)*Tc
    tn = t/(0.2+0.15*Tc)
    e_t = (Emax-Emin)*1.55*(tn/alpha1)**n1/((tn/alpha1)**n1+1)*1/((tn/alpha2)**n2+1) + Emin
    dydt = [0, 1/(R1*C2)*(-x2+x3), 1/(R1*C3)*(x2-x3), 0]
    
    if (t>0.05):
      if (x2 <= x1*e_t and x1*e_t >= x3): #ejection x1*e_t >= x3
        dydt[0] += -x4
        dydt[2] += x4/C3
        dydt[3] += e_t/L*x1 - x3/L - (R3+R4)*x4/L
      else: #filling
        if (x2 >= x1*e_t and x1*e_t <= x3):
          dydt[0] += 1/R2*(-x1*e_t+x2)
          dydt[1] += e_t/(R2*C2)*x1 - x2/(R2*C2)

    return dydt

  def simulator(self):

    N = 100

    Tc = self.parameters[0]
    Emax = self.parameters[8]
    Emin = self.parameters[9]
    n1 = self.parameters[10]
    n2 = self.parameters[11]
    alpha1 = self.parameters[12]
    alpha2 = self.parameters[13]
    Vd = self.parameters[-3]
    start_v = self.parameters[-2]
    start_pa = self.parameters[-1]

    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig4b in ferreira2005nonlinear
    start_pla = start_v*Emin #self.Elastance(0, Tc, Emax, Emin, n1, n2, alpha1, alpha2) #start_pla = start_plv at t=0 (see Fig4a in ferreira2005nonlinear)

    y0 = [start_v, start_pla, start_pa, start_qt]

    t = np.linspace(0, Tc*N, int(60000*N)) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle

    sol = odeint(self.heart_ode, y0, t, args = (self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3], self.parameters[4], self.parameters[5], self.parameters[6], self.parameters[7], self.parameters[8],self.parameters[9], self.parameters[10], self.parameters[11], self.parameters[12], self.parameters[13])) #t: list of values

    result_Vlv = np.array(sol[:, 0]) + Vd
    #get Plv from Vlv-Vd
    result_Plv = np.array([self.Elastance(x, Tc, Emax, Emin, n1, n2, alpha1, alpha2)*v for x,v in zip(t,sol[:, 0])])
    result_Pa = np.array(sol[:, 1])

    ved = max(result_Vlv[int(N-5)*60000:int(N)*60000])
    ves = min(result_Vlv[int(N-5)*60000:int(N)*60000])

    pes = max(result_Plv[int(N-5)*60000:int(N)*60000])
    ped = min(result_Plv[int(N-5)*60000:int(N)*60000])

    plt.plot(t[int(N-5)*60000:int(N)*60000], result_Vlv[int(N-5)*60000:int(N)*60000])
    plt.axvline(x=Tc*(N-3), color='r')
    plt.axvline(x=0.5*Tc+Tc*(N-3), color='g')
    plt.show()
    plt.plot(result_Vlv[int(N-5)*60000:int(N)*60000], result_Plv[int(N-5)*60000:int(N)*60000], color='red')
    plt.show()

    return ved, ves, ped, pes
  
  
 class stergiopoulos1996determinants: 
  def __init__(self):
    self.n_parameters = 15
    self.name_parameters = ['Tc', 'Pv', 'Rc', 'Rp', 'Rv', 'C', 'Emax', 'Emin', 'n1', 'n2', 'alpha1', 'alpha2', 'Vd', 'start_v', 'start_pa']
    self.parameters = [0.8, 7.5, 0.1, 1.05, 0.005, 1.60, 2.31, 0.06, 1.32, 21.9, 0.303, 0.508, 11., 140., 80.]
    self.validity_intervals = [[0.5, 2.], [7., 12.], [0.05, 0.8], [0.7, 6], [0.0001, 0.25], [1., 3.], [0.5, 6.], [0.01, 0.1], [1., 4.], [10., 35.], [0.303, 0.303], [0.508, 0.508], [4., 40.], [-20., 250.], [60., 80.]]
    self.learnable_parameters = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]

  def relu(x):
    if (x>0): return x
    return 0
  
  def Elastance(self, t, Tc, Emax, Emin, n1, n2, alpha1, alpha2):
    #can remove this if only want 1st ED (and the 1st ES before)
    tn = t-int(t/Tc)*Tc
    et = Emax*(tn/(alpha1*Tc))**n1/((tn/(alpha1*Tc))**n1+1)*1/((tn/(alpha2*Tc))**n2+1) + Emin

    return et

  def heart_ode(self, y, t, Tc, Pv, Rc, Rp, Rv, C, Emax, Emin, n1, n2, alpha1, alpha2):
    #Tc, Pv, Rc, Rp, Rv, C, Emax, Emin, n1, n2, alpha1, alpha2 = pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7], pars[8], pars[9], pars[10], pars[11]

    x1, x2 = y #here y is a vector of 4 values (not functions), at time t, used for getting (dy/dt)(t)
    
    tn = t-int(t/Tc)*Tc
    et = Emax*(tn/(alpha1*Tc))**n1/((tn/(alpha1*Tc))**n1+1)*1/((tn/(alpha2*Tc))**n2+1) + Emin

    dydt = [ max(Pv - x1*et,0)/Rv - max(x1*et - x2,0)/Rc, max(x1*et-x2,0)/(Rc*C) - x2/(Rp*C) ]

    return dydt

  def simulator(self):

      N = 100

      Tc = self.parameters[0]
      Emax = self.parameters[6]
      Emin = self.parameters[7]
      n1 = self.parameters[8]
      n2 = self.parameters[9]
      alpha1 = self.parameters[10]
      alpha2 = self.parameters[11]
      Vd = self.parameters[-3]

      start_v = self.parameters[-2]
      start_pa = self.parameters[-1]
      
      y0 = [start_v, start_pa]

      t = np.linspace(0, Tc*N, int(60000*N)) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle

      sol = odeint(self.heart_ode, y0, t, args = (self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3], self.parameters[4], self.parameters[5], self.parameters[6], self.parameters[7], self.parameters[8], self.parameters[9], self.parameters[10], self.parameters[11])) #t: list of values
      
      result_Vlv = np.array(sol[:, 0]) + Vd
      #get Plv from Vlv-Vd
      result_Plv = np.array([self.Elastance(x, Tc, Emax, Emin, n1, n2, alpha1, alpha2)*v for x,v in zip(t,sol[:, 0])])
      result_Pa = np.array(sol[:, 1])

      ved = max(result_Vlv[int(N-5)*60000:int(N)*60000])
      ves = min(result_Vlv[int(N-5)*60000:int(N)*60000])

      pes = max(result_Plv[int(N-5)*60000:int(N)*60000])
      ped = min(result_Plv[int(N-5)*60000:int(N)*60000])

      plt.plot(t[int(N-5)*60000:int(N)*60000], result_Vlv[int(N-5)*60000:int(N)*60000])
      plt.axvline(x=Tc*(N-3), color='r')
      plt.axvline(x=0.5*Tc+Tc*(N-3), color='g')
      plt.show()
      plt.plot(result_Vlv[int(N-5)*60000:int(N)*60000], result_Plv[int(N-5)*60000:int(N)*60000], color='red')
      plt.show()
      
      return ved, ves, ped, pes
    

class her2018windkessel:
  def __init__(self):
    self.n_parameters = 20
    self.name_parameters = ['Tc', 'Pla', 'Rmvo', 'Rmvc', 'Ravo', 'Ravc', 'Rao', 'Cao', 'Lao', 'Cpa', 'Rpa', 'Emax', 'Emin', 'n1', 'n2', 'alpha1', 'alpha2', 'Vd', 'start_v', 'start_pa']
    self.parameters = [0.8, 10., 0.01, 1000., 0.002, 1000., 0.08, 0.15, 0.0015, 1.5, 1.35, 1.52, 0.08, 1.9, 21.9, 0.7, 1.17, 10., 140., 80.]
    self.validity_intervals = [[0.5, 2.], [7., 12.], [0.0001,0.1], [100, 100000], [0.0001, 0.1], [100, 100000], [0.07, 0.2], [0.0001, 0.3], [0.0000015, 0.005], [0.8, 4.], [0.8, 10.], [0.5, 4.5], [0.01, 0.3], [1., 4.], [10., 35.], [0.7, 0.7], [1.17, 1.17], [4., 40.], [5., 350.], [60., 90.]]
    self.learnable_parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]

  def Elastance(self, t, Tc, Emax, Emin, n1, n2, alpha1, alpha2):
    tn = (t-int(t/Tc)*Tc)/(0.2+0.15*Tc)
    et = (Emax-Emin)*1.55*(tn/(alpha1))**n1/((tn/(alpha1))**n1+1)*1/((tn/(alpha2))**n2+1) + Emin
    
    return et

  def heart_ode(self, y, t, Tc, Pla, Rmvo, Rmvc, Ravo, Ravc, Rao, Cao, Lao, Cpa, Rpa, Emax, Emin, n1, n2, alpha1, alpha2):
      x1, x2, x3, x4 = y #here y is a vector of 4 values (not functions), at time t, used for getting (dy/dt)(t)
      tn = (t-int(t/Tc)*Tc)/(0.2+0.15*Tc)
      e_t = (Emax-Emin)*1.55*(tn/(alpha1))**n1/((tn/(alpha1))**n1+1)*1/((tn/(alpha2))**n2+1) + Emin
    
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

  def simulator(self):

    N = 100

    Tc = self.parameters[0]
    Emax = self.parameters[11]
    Emin = self.parameters[12]
    n1 = self.parameters[13]
    n2 = self.parameters[14]
    alpha1 = self.parameters[15]
    alpha2 = self.parameters[16]
    Vd = self.parameters[17]
    start_v = self.parameters[18]
    start_pa = self.parameters[19]

    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig4b in ferreira2005nonlinear
    start_pla = start_v*Emin #self.Elastance(0, Tc, Emax, Emin, n1, n2, alpha1, alpha2) #start_pla = start_plv at t=0 (see Fig4a in ferreira2005nonlinear)

    y0 = [start_v, start_pla, start_pa, start_qt]

    t = np.linspace(0, Tc*N, int(60000*N)) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle

    sol = odeint(self.heart_ode, y0, t, args = (self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3], self.parameters[4], self.parameters[5], self.parameters[6], self.parameters[7], self.parameters[8],self.parameters[9], self.parameters[10], self.parameters[11], self.parameters[12], self.parameters[13], self.parameters[14], self.parameters[15], self.parameters[16])) #t: list of values

    result_Vlv = np.array(sol[:, 0]) + Vd
    #get Plv from Vlv-Vd
    result_Plv = np.array([self.Elastance(x, Tc, Emax, Emin, n1, n2, alpha1, alpha2)*v for x,v in zip(t,sol[:, 0])])
    result_Pa = np.array(sol[:, 1])

    ved = max(result_Vlv[int(N-5)*60000:int(N)*60000])
    ves = min(result_Vlv[int(N-5)*60000:int(N)*60000])

    pes = max(result_Plv[int(N-5)*60000:int(N)*60000])
    ped = min(result_Plv[int(N-5)*60000:int(N)*60000])

    plt.plot(t[int(N-5)*60000:int(N)*60000], result_Vlv[int(N-5)*60000:int(N)*60000])
    plt.axvline(x=Tc*(N-3), color='r')
    plt.axvline(x=0.5*Tc+Tc*(N-3), color='g')
    plt.show()
    plt.plot(result_Vlv[int(N-5)*60000:int(N)*60000], result_Plv[int(N-5)*60000:int(N)*60000], color='red')
    plt.show()

    return ved, ves, ped, pes
  
  
## example of use:

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy import interpolate
from matplotlib import pyplot
import sys
import matplotlib.pyplot as plt

model1 = her2018windkessel()
ved, ves, ped, pes = model1.simulator()
