##########################################################################
# PINN for lumped parameter cardiac analogy model
# Author: Franny Dean
# Date: 2024-04-18
##########################################################################

import os
print('Current working directory:', os.getcwd())
import time
import datetime
import tensorflow as tf
import numpy as np
import math
import torch
from math import prod
import itertools
import random
from scipy.integrate import odeint 
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt

##########################################################################
# PINN module
# built from Shota DEGUCHI, Yosuke SHIBATA Structural Analysis Laboratory, Kyushu University (Jul. 19th, 2021) implementation of PINN - Physics-Informed Neural Network on TensorFlow 2

class PINN:
    def __init__(self, 
                 t, x1_train,
                 in_dim, out_dim, width, depth, 
                 w_init = "Glorot", b_init = "zeros", act = "tanh", 
                 lr = 1e-3, opt = "Adam", 
                 f_scl = "minmax", laaf = False, inv = True, 
                 Tc = 1., Vd = 1., Emax = 1., Emin = 1., Rm = 1., Ra = .1,
                 f_mntr = 10, r_seed = 1234):
        
        # initialize the configuration
        self.r_seed = r_seed
        self.random_seed(seed = self.r_seed)
        self.dat_typ = tf.float64
        self.in_dim  = in_dim    # input dimension
        self.out_dim = out_dim   # output dimension
        self.width   = width     # middle dimension
        self.depth   = depth     # (# of hidden layers) + output layer
        self.w_init  = w_init    # weight initializer
        self.b_init  = b_init    # bias initializer
        self.act     = act       # activation function
        self.lr      = lr        # learning rate
        self.opt     = opt       # optimizer (SGD, RMSprop, Adam, etc.)
        self.f_scl   = f_scl     # feature scaling
        self.laaf    = laaf      # LAAF? (L-LAAF, GAAF / N-LAAF not implemented)
        self.inv     = inv       # inverse problem? 
        self.f_mntr  = f_mntr    # monitoring frequency
        
        # ODE
        self.x1 = x1_train; self.t = t
        
        # bounds (for feature scaling)
        bounds  = t
        self.lb = tf.cast(tf.reduce_min (bounds, axis = 0), self.dat_typ)
        self.ub = tf.cast(tf.reduce_max (bounds, axis = 0), self.dat_typ)
        self.mn = tf.cast(tf.reduce_mean(bounds, axis = 0), self.dat_typ)

        # build a network
        self.structure = [self.in_dim] + [self.width] * (self.depth - 1) + [self.out_dim]
        self.weights, self.biases, self.alphas, self.params = self.dnn_init(self.structure)
        
        # system params
        self.Tc = Tc
        self.Vd = Vd
        self.Emax = Emax
        self.Emin = Emin
        self.Rm = Rm
        self.Ra = Ra
        if self.inv == True:
            self.Tc = tf.Variable(self.Tc, dtype = self.dat_typ)
            self.Vd = tf.Variable(self.Vd, dtype = self.dat_typ)
            self.Emax = tf.Variable(self.Emax, dtype = self.dat_typ)
            self.Emin = tf.Variable(self.Emin, dtype = self.dat_typ)
            self.Rm = tf.Variable(self.Rm, dtype = self.dat_typ)
            self.Ra  = tf.Variable(self.Ra , dtype = self.dat_typ)
            self.params.append(self.Tc)
            self.params.append(self.Vd)
            self.params.append(self.Emax)
            self.params.append(self.Emin)
            self.params.append(self.Rm)
            self.params.append(self.Ra)
            self.Tc_log = []
            self.Vd_log  = []
            self.Emax_log  = []
            self.Emin_log  = []
            self.Rm_log  = []
            self.Ra_log  = []
        elif self.inv == False:
            self.Tc = tf.constant(self.Tc, dtype = self.dat_typ)
            self.Vd  = tf.constant(self.Vd , dtype = self.dat_typ)
            self.Emax  = tf.constant(self.Emax , dtype = self.dat_typ)
            self.Emin  = tf.constant(self.Emin , dtype = self.dat_typ)
            self.Rm  = tf.constant(self.Rm , dtype = self.dat_typ)
            self.Ra  = tf.constant(self.Ra , dtype = self.dat_typ)
        else:
            raise NotImplementedError(">>>>> system params")

        # optimization
        self.optimizer    = self.opt_(self.lr, self.opt)
        self.ep_log       = []
        self.loss_glb_log = []
        self.loss_pde_log = []
        self.loss_dat_log = []
        
        print("\n************************************************************")
        print("****************     MAIN PROGRAM START     ****************")
        print("************************************************************")
        print(">>>>> start time:", datetime.datetime.now())
        print(">>>>> configuration;")
        print("         random seed  :", self.r_seed)
        print("         data type    :", self.dat_typ)
        print("         activation   :", self.act)
        print("         weight init  :", self.w_init)
        print("         bias   init  :", self.b_init)
        print("         learning rate:", self.lr)
        print("         optimizer    :", self.opt)
        print("         width        :", self.width)
        print("         depth        :", self.depth)
        print("         structure    :", self.structure)
        
    def random_seed(self, seed = 1234):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
    def dnn_init(self, strc):
        weights = []
        biases  = []
        alphas  = []
        params  = []
        for d in range(0, self.depth):   # depth = self.depth
            w = self.weight_init(shape = [strc[d], strc[d + 1]], depth = d)
            b = self.bias_init  (shape = [      1, strc[d + 1]], depth = d)
            weights.append(w)
            biases .append(b)
            params .append(w)
            params .append(b)
            if self.laaf == True and d < self.depth - 1:
                a = tf.Variable(1., dtype = self.dat_typ, name = "a" + str(d))
                params.append(a)
            else:
                a = tf.constant(1., dtype = self.dat_typ)
            alphas .append(a)
        return weights, biases, alphas, params
        
    def weight_init(self, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if self.w_init == "Glorot":
            std = np.sqrt(2 / (in_dim + out_dim))
        elif self.w_init == "He":
            std = np.sqrt(2 / in_dim)
        elif self.w_init == "LeCun":
            std = np.sqrt(1 / in_dim)
        else:
            raise NotImplementedError(">>>>> weight_init")
        weight = tf.Variable(
            tf.random.truncated_normal(shape = [in_dim, out_dim], \
            mean = 0., stddev = std, dtype = self.dat_typ), \
            dtype = self.dat_typ, name = "w" + str(depth)
            )
        return weight
    
    def bias_init(self, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if self.b_init == "zeros":
            bias = tf.Variable(
                tf.zeros(shape = [in_dim, out_dim], dtype = self.dat_typ), \
                dtype = self.dat_typ, name = "b" + str(depth)
                )
        elif self.b_init == "ones":
            bias = tf.Variable(
                tf.ones(shape = [in_dim, out_dim], dtype = self.dat_typ), \
                dtype = self.dat_typ, name = "b" + str(depth)
                )
        else:
            raise NotImplementedError(">>>>> bias_init")
        return bias
    
    def opt_(self, lr, opt):
        if opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate = lr, momentum = 0.0, nesterov = False
                )
        elif opt == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate = lr, rho = 0.9, momentum = 0.0, centered = False
                )
        elif opt == "Adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False
                )
        elif opt == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(
                learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999
                )
        elif opt == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(
                learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999
                )
        else:
            raise NotImplementedError(">>>>> opt_")
        return optimizer
    
    def forward_pass(self, x):
        # feature scaling
        if self.f_scl == "minmax":
            z = 2. * (x - self.lb) / (self.ub - self.lb) - 1.
        elif self.f_scl == "mean":
            z = (x - self.mn) / (self.ub - self.lb)
        else:
            raise NotImplementedError(">>>>> forward_pass (f_scl)")
        # forward pass
        for d in range(0, self.depth - 1):
            w = self.weights[d]
            b = self.biases [d]
            a = self.alphas [d]
            u = tf.add(tf.matmul(z, w), b)
            u = tf.multiply(a, u)
            if self.act == "tanh":
                z = tf.tanh(u)
            elif self.act == "swish":
                z = tf.multiply(u, tf.sigmoid(u))
            elif self.act == "gelu":
                z = tf.multiply(u, tf.sigmoid(1.702 * u))
            elif self.act == "mish":
                z = tf.multiply(u, tf.tanh(tf.nn.softplus(u)))
            else:
                raise NotImplementedError(">>>>> forward_pass (act)")
        w = self.weights[-1]
        b = self.biases [-1]
        a = self.alphas [-1]
        u = tf.add(tf.matmul(z, w), b)
        u = tf.multiply(a, u)
        z = u   # identity mapping
        y = z
        return y
        
    def pde(self, t):
        t = tf.convert_to_tensor(t, dtype = self.dat_typ)
          
        Tc = self.Tc 
        Vd = self.Vd
        Emax = self.Emax 
        Emin = self.Emin 
        Rm = self.Rm 
        Ra = self.Ra 
        
        # other params
        Rc=float(0.0398)
        Rs=float(1.0000)
        Ca=float(0.0800)
        Cs=float(1.3300)
        Cr=float(4.400)
        Ls=float(0.0005)
        
        with tf.GradientTape(persistent = True) as tp:
            tp.watch(t)
            outputs = self.forward_pass(t)
            x1 = outputs[:,0:1]
            x2 = outputs[:,1:2]
            x3 = outputs[:,2:3]
            x4 = outputs[:,3:4]
            x5 = outputs[:,4:5]

        
            x1dt = tp.gradient(x1, t)[0]
            x2dt = tp.gradient(x2, t)[0]
            x3dt = tp.gradient(x3, t)[0]
            x4dt = tp.gradient(x4, t)[0]
            x5dt = tp.gradient(x5, t)[0]
        
        P_lv = self.Plv(x1, Emax, Emin, t, Tc)
        
        f_1 = tf.math.subtract(tf.map_fn(r, tf.math.subtract(x2,P_lv))/Rm-tf.map_fn(r, tf.math.subtract(P_lv,x4))/Ra, x1dt)
        f_2 = tf.math.subtract((tf.math.subtract(x3,x2))/(Rs*Cr)-tf.map_fn(r, tf.math.subtract(x2,P_lv))/(Cr*Rm), x2dt)
        f_3 = tf.math.subtract((tf.math.subtract(x2,x3))/(Rs*Cs)+x5/Cs, x3dt)
        f_4 = tf.math.subtract(-x5/Ca+ tf.map_fn(r, tf.math.subtract(P_lv,x4))/(Ca*Ra), x4dt)
        f_5 = tf.math.subtract((tf.math.subtract(tf.math.subtract(x4,x3),Rc*x5))/Ls, x5dt)
        
        del tp
        return x1, x2, x3, x4, x5, f_1, f_2, f_3, f_4, f_5

    def loss_dat(self, x1, t):
        x1_,_,_,_,_,_,_,_,_,_ = self.pde(t)
        loss =   tf.reduce_mean(tf.square(x1 - x1_))
        return loss
        
    def loss_pde(self, t):
        _,_,_,_,_, f_1_, f_2_, f_3_, f_4_, f_5_ = self.pde(t)
        loss =   tf.reduce_mean(tf.square(f_1_)) \
               + tf.reduce_mean(tf.square(f_2_)) \
               + tf.reduce_mean(tf.square(f_3_)) \
               + tf.reduce_mean(tf.square(f_4_)) \
               + tf.reduce_mean(tf.square(f_5_))
        return loss
    
    @tf.function
    def loss_glb(self, x1, 
                 t):
        loss =   self.loss_dat(x1, t) \
               + self.loss_pde(t)
        if self.laaf == True:
            loss += 1. / tf.reduce_mean(tf.exp(self.alphas))
        else:
            pass
        return loss

    def loss_grad(self, 
                  x1, t):
        with tf.GradientTape(persistent = True) as tp:
            loss = self.loss_glb(x1, t)
        grad = tp.gradient(loss, self.params)
        del tp
        #print(loss, grad)
        return loss, grad
    
    @tf.function
    def grad_desc(self, 
                  x1, t):
        loss, grad = self.loss_grad(x1, t)
        self.optimizer.apply_gradients(zip(grad, self.params))
        return loss
        
    def train(self, epoch = int(1e3), batch = 0, tol = 1e-5): # batch = 2 ** 6?
        print(">>>>> training setting;")
        print("         # of epoch     :", epoch)
        print("         batch size     :", batch)
        print("         convergence tol:", tol)

        # boundary & PDE
        x1 = self.x1; t = self.t

        t0 = time.time()
        for ep in range(epoch):
            ep_loss = 0 # this needs changed....
            if batch == 0:
                ep_loss = self.grad_desc(
                    x1, t)
            else:
                raise NotImplementedError(">>>>> BATCH NOT IMPLEMENTED YET THERE IS NO ep_locc UPDATING")
                n_b = x1.shape[0]
                idx_b = np.random.permutation(n_b)

                for idx in range(0, n_b, batch):
                    x1_b = x1[idx_b[idx:idx+batch if idx+batch<n_b else n_b]]


            if ep % self.f_mntr == 0:
                elps = time.time() - t0

                loss_dat = self.loss_dat(x1, t)
                loss_pde = self.loss_pde(t)

                self.ep_log.append(ep)
                self.loss_glb_log.append(ep_loss)
                self.loss_pde_log.append(loss_pde)
                self.loss_dat_log.append(loss_dat)
                
                if self.inv == True:
                    ep_Tc = self.Tc.numpy()
                    ep_Vd  = self.Vd .numpy()
                    ep_Emax = self.Emax.numpy()
                    ep_Emin = self.Emin.numpy()
                    ep_Rm = self.Rm.numpy()
                    ep_Ra = self.Ra.numpy()
                    self.Tc_log.append(ep_Tc)
                    self.Vd_log .append(ep_Vd)
                    self.Emax_log.append(ep_Emax)
                    self.Emin_log.append(ep_Emin)
                    self.Rm_log.append(ep_Rm)
                    self.Ra_log.append(ep_Ra)
                    print(f"ep: {ep}, loss: {ep_loss}, loss_pde: {loss_pde}, Tc: {ep_Tc}, Vd: {ep_Vd}, Emax: {ep_Emax}, Emin {ep_Emin}, Rm: {ep_Rm}, Ra: {ep_Ra}, elps: {elps}")
                elif self.inv == False:
                    print(f"ep: {ep}, loss: {ep_loss}, loss_pde: {loss_pde}, elps: {elps}, loss_dat: {loss_dat}")
                    
                else:
                    raise NotImplementedError(">>>>> system params")
                t0 = time.time()
            
            if ep_loss < tol:
                print(">>>>> program terminating with the loss converging to its tolerance.")
                print("\n************************************************************")
                print("*****************     MAIN PROGRAM END     *****************")
                print("************************************************************")
                print(">>>>> end time:", datetime.datetime.now())
                break
        
        print("\n************************************************************")
        print("*****************     MAIN PROGRAM END     *****************")
        print("************************************************************")
        print(">>>>> end time:", datetime.datetime.now())
                
    def inference(self, t):
        x1_, x2_, x3_, x4_, x5_, f_1_, f_2_, f_3_, f_4_, f_5_ = self.pde(t)
        return x1_, x2_, x3_, x4_, x5_, f_1_, f_2_, f_3_, f_4_, f_5_
    
    #returns Elastance(t)
    def Elastance(self, t):
        t = t- tf.map_fn(fl,(t/self.Tc))*self.Tc
        tn = t/0.2+0.15*self.Tc
        return (self.Emax - self.Emin)*1.55*(tn/0.7)**1.9/((tn/0.7)**1.9+1)*1/((tn/1.17)**21.9+1) + self.Emin
        
    #returns Plv at time t using Elastance(t) and Vlv(t)-Vd=x1
    def Plv(self, x1, Emax, Emin, t, Tc):
        return tf.map_fn(self.Elastance, t)*x1
        
@tf.function
def r(u):
    if u<0.:
        return tf.zeros((1,),dtype=tf.float64)
    else:
        return u
    
@tf.function
def fl(u):
    return tf.cast(tf.math.floor(u),dtype=tf.float64)

##########################################################################
# functions for synthetic data generation
def heart_ode(y, t, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc):
    x1, x2, x3, x4, x5 = y #here y is a vector of 5 values (not functions), at time t, used for getting (dy/dt)(t)
    P_lv = Plv2(x1,Emax,Emin,t,Tc)
    dydt = [r2(x2-P_lv)/Rm-r2(P_lv-x4)/Ra, (x3-x2)/(Rs*Cr)-r2(x2-P_lv)/(Cr*Rm), (x2-x3)/(Rs*Cs)+x5/Cs, -x5/Ca+r2(P_lv-x4)/(Ca*Ra), (x4-x3-Rc*x5)/Ls]
    return dydt

def r2(u):
    if u<0.0:
        return 0.0
    else:
        return u

#returns Plv at time t using Elastance(t) and Vlv(t)-Vd=x1
def Plv2(x1,Emax,Emin,t, Tc):
    return Elastance2(Emax,Emin,t, Tc)*x1

#returns Elastance(t)
def Elastance2(Emax,Emin,t, Tc):
    t = t-int(t/Tc)*Tc #can remove this if only want 1st ED (and the 1st ES before)
    tn = t/(0.2+0.15*Tc)
    return (Emax-Emin)*1.55*(tn/0.7)**1.9/((tn/0.7)**1.9+1)*1/((tn/1.17)**21.9+1) + Emin
     
# Define your function here (for example, a 2-variable function)
def f(Tc, start_v, Emax, Emin, Rm, Ra, Vd, N, plotloops):
    startp = 75.
    Rs = 1.0
    Rc = 0.0398
    Ca = 0.08
    Cs = 1.33
    Cr = 4.400
    Ls = 0.0005

    start_pla = float(start_v*Elastance2(Emax, Emin, 0, Tc))
    start_pao = startp
    start_pa = start_pao
    start_qt = 0 #aortic flow is Q_T and is 0 at ED, also see Fig5 in simaan2008dynamical
    y0 = [start_v, start_pla, start_pa, start_pao, start_qt]

    t = np.linspace(0, Tc*N, int(60000*N)) #spaced numbers over interval (start, stop, number_of_steps), 60000 time instances for each heart cycle
    #changed to 60000 for having integer positions for Tmax

    sol = odeint(heart_ode, y0, t, args = (Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc)) #t: list of values

    result_Vlv = np.array(sol[:, 0]) + Vd
    result_Plv = np.array([Plv2(v, Emax, Emin, xi, Tc) for xi,v in zip(t,sol[:, 0])])

    ved = sol[(N-1)*60000, 0] + Vd
    ves = sol[200*int(60/Tc)+9000+(N-1)*60000, 0] + Vd
    ef = (ved-ves)/ved * 100.

    minv = min(result_Vlv[(N-1)*60000:N*60000-1])
    maxv = max(result_Vlv[(N-1)*60000:N*60000-1])
    minp = min(result_Plv[(N-1)*60000:N*60000-1])
    maxp = max(result_Plv[(N-1)*60000:N*60000-1])

    ved2 = sol[(N-1)*60000 - 1, 0] + Vd
    isperiodic = 0
    if (abs(ved-ved2) > 5.): isperiodic = 1

    if plotloops:
      plt.plot(result_Vlv[(N-2)*60000:(N)*60000], result_Plv[(N-2)*60000:N*60000])
      plt.xlabel("LV volume (ml)")
      plt.ylabel("LV pressure (mmHg)")
      plt.show()

    # save volumes to test PINN and SSM
    volumes = result_Vlv[(N-1)*60000:N*60000-1]

    return ved, ves, ef, minv, maxv, minp, maxp, volumes, isperiodic

# generate synthetic data
def generate_synthetic_data(N=70):
    output_path = 'Desktop/Physics_Informed_Transfer_Learning/CardioPINN_for_modeling/exploratory_experiments/final_model_weights/'

    ints = [[0.4, 1.7], [0., 280.], [0.5, 3.5], [0.02, 0.1], [0.005, 0.1], [0.0001, 0.25]] #validity interval for each learnable parameter
    nvals_pars = [4, 4, 4, 4, 4, 4] #number of values taken from each parameter
    vds = np.linspace(4., 25., 15) # Vd volumes taken (not used for the interpolator)

    n_pars = len(ints)
    n_points = prod(nvals_pars)

    pars = []
    for i in range(n_pars):
        pars.append(np.linspace(ints[i][0], ints[i][1], nvals_pars[i]))

    points = list(itertools.product(*pars))

    veds = []
    vess = []
    used_x = []
    volume_time_series = []

    selected_rows = []
    indices = np.random.choice(n_points,size=1000, replace=False)
    for i in indices:
        selected_rows.append(points[i])

    for point in selected_rows:
        ved, ves, ef, minv, maxv, minp, maxp, volumes, isperiodic = f(*point, vds[0], N, False) ##recommended to plot the pv loops when building the intervals
        veds.append(maxv)
        vess.append(minv)
        used_x.append(point)
        volume_time_series.append(volumes)

    #convert into torch tensors:
    pts = torch.tensor(used_x)
    tensor1 = torch.tensor(veds)
    tensor2 = torch.tensor(vess)
    vedves = torch.stack((tensor1, tensor2), dim=1)
    full_volumes = torch.tensor(volume_time_series)

    #save points and ved, ves:
    file = f'points_{date.today()}'
    torch.save(pts, os.path.join(output_path,f'{file}.pt'))
    file = f'vedves_{date.today()}'
    torch.save(vedves, os.path.join(output_path,f'{file}.pt'))
    file = f'volumes_{date.today()}'
    torch.save(full_volumes, os.path.join(output_path,f'{file}.pt'))
    print("Saved")

##########################################################################


def main(iterations=1, generate_data=False, data_date=date.today(), N=70, toler = 3, epochs = 1):

    if generate_data:
        generate_synthetic_data(N=N)

    print('Done generating data')
    
    # load data
    output_path = 'Desktop/Physics_Informed_Transfer_Learning/CardioPINN_for_modeling/exploratory_experiments/final_model_weights/'
    used_x = torch.load(os.path.join(output_path,f'points_{data_date}.pt'))
    vedves = torch.load(os.path.join(output_path,f'vedves_{data_date}.pt'))
    volume_time_series = torch.load(os.path.join(output_path,f'volumes_{data_date}.pt'))


    if iterations==1:  
        print('Testing one parameter set')
        # pick a random point
        random = np.random.randint(0, len(used_x))

        # ground truth EF
        ef = (vedves[random][0] - vedves[random][1]) / vedves[random][0] * 100.

        # pick a sample of len(volume_time_series[random])/step points
        sample = np.arange(0, len(volume_time_series[random]),step =20)

        # training data (synthetic sample)
        x1_train = volume_time_series[random][sample].transpose(0,-1).reshape((len(sample), 1))
        t_train = np.arange((N-1)*60000, N*60000-1, 1)[sample].transpose().reshape((len(sample), 1))
        t_train = t_train.astype(np.float64)
        
        # print the parameter set used
        print(f'Tc= {used_x[random][0]}, Vd = 4., Emax = {used_x[random][2]}, Emin = {used_x[random][3]}, Rm = {used_x[random][4]}, Ra = {used_x[random][5]}')

        # PINN
        pinn = PINN(t_train, x1_train, in_dim=1, out_dim=5, width = 2 ** 8, depth = 5, inv=False,
                    Tc = used_x[random][0], Vd = 4., Emax = used_x[random][2], Emin = used_x[random][3], Rm = used_x[random][4], Ra = used_x[random][5])

        # Training
        pinn.train(tol=toler, epoch=epochs, batch=0)

        # validation/inference
        x1_, x2_, x3_, x4_, x5_, f_1_, f_2_, f_3_, f_4_, f_5_ = pinn.inference(t_train)
        diff_mean = tf.reduce_mean(x1_train - x1_)
        print(f"Mean abs error: {diff_mean}")

        # to calculate EF
        maxv_ = max(x1_+4.) # add Vd
        minv_ = min(x1_+4.) # add Vd
        print(f'Predicted EF: {(maxv_-minv_)/maxv_*100}, Ground Truth EF: EF {ef}')

        results = pd.DataFrame()
        # save results and parameter set
        results = results._append({'i': random, 'Tc': used_x[random][0], 'Vd': 4., 'Emax': used_x[random][2], 'Emin': used_x[random][3], 'Rm': used_x[random][4], 'Ra': used_x[random][5], 'EF_pred': (maxv_-minv_)/maxv_*100, 'EF_true': ef, 'MAE_volumes': diff_mean}, ignore_index=True)
        file = f'single_results_{date.today()}_{datetime.datetime.now().time()}'
        results.to_csv(os.path.join(output_path,f'{file}.csv'))
        print("Saved results of single iteration")

    else:
        print('Starting testing of all parameter sets')
        # PINN
        results = pd.DataFrame()
        
        # check length
        if iterations > len(used_x):
            print('Number of iterations exceeds number of parameter sets instead using {len(used_x)} iterations')
            iterations = len(used_x)

        for i in range(iterations):
            
            # ground truth EF
            ef = (vedves[i][0] - vedves[i][1]) / vedves[i][0] * 100.

            # pick a sample of len(volume_time_series[i])/step points
            sample = np.arange(0, len(volume_time_series[i]),step =20)

            # training data (synthetic sample)
            x1_train = volume_time_series[i][sample].transpose(0,-1).reshape((len(sample), 1))
            print(x1_train.shape)
            t_train = np.arange((N-1)*60000, N*60000-1, 1)[sample].transpose().reshape((len(sample), 1))
            t_train = t_train.astype(np.float64)
            
            # forward process not inverse
            # order of parameters in pts: Tc 0, start_v 1, Emax 2, Emin 3, Rm 4, Ra 5, Vd 6
            print(f'Tc= {used_x[i][0]}, Vd = 4., Emax = {used_x[i][2]}, Emin = {used_x[i][3]}, Rm = {used_x[i][4]}, Ra = {used_x[i][5]}')
            pinn = PINN(t_train, x1_train, in_dim=1, out_dim=5, width=2 ** 8, depth=5, inv=False,
                        Tc=used_x[i][0], Vd=4., Emax=used_x[i][2], Emin=used_x[i][3], Rm=used_x[i][4], Ra=used_x[i][5])

            # Training
            pinn.train(tol=toler, epoch=epochs, batch=0)

            # validation/inference
            x1_, x2_, x3_, x4_, x5_, f_1_, f_2_, f_3_, f_4_, f_5_ = pinn.inference(t_train)
            diff_mean = tf.reduce_mean(x1_train - x1_)
            print(f"Iteration: {i}, Sample size: {len(sample)}, Mean abs error: {diff_mean}")
            # for EF
            maxv_ = max(x1_+4.) # add Vd
            minv_ = min(x1_+4.) # add Vd
            print(f'Predicted EF: {(maxv_-minv_)/maxv_*100}, Ground Truth EF: EF {ef}')

            # save results and parameter set
            results = results._append({'i': i, 'Tc': used_x[i][0], 'Vd': 4., 'Emax': used_x[i][2], 'Emin': used_x[i][3], 'Rm': used_x[i][4], 'Ra': used_x[i][5], 'EF_pred': (maxv_-minv_)/maxv_*100, 'EF_true': ef, 'MAE_volumes': diff_mean}, ignore_index=True)

        file = f'results_{date.today()}_{datetime.datetime.now().time()}'
        results.to_csv(os.path.join(output_path,f'{file}.csv'))
        print("Saved results of all iterations")
    
if __name__ == "__main__":
  
    main(iterations=1, generate_data=True, data_date=date.today(), N=70, toler = 3, epochs = 50000)

