##########################################################################
# Neural ODE Baseline for lumped parameter cardiac analogy model
##########################################################################
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from torchdiffeq import odeint_adjoint as odeint
from scipy.integrate import odeint as odeint_standard

# Define ODE functions and simulator
def heart_ode(y, t, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc):
    x1, x2, x3, x4, x5 = y
    P_lv = Plv(x1, Emax, Emin, t, Tc)
    dydt = [r(x2-P_lv)/Rm-r(P_lv-x4)/Ra, (x3-x2)/(Rs*Cr)-r(x2-P_lv)/(Cr*Rm), (x2-x3)/(Rs*Cs)+x5/Cs, -x5/Ca+r(P_lv-x4)/(Ca*Ra), (x4-x3-Rc*x5)/Ls]
    return dydt

def r(u):
    return max(u, 0.)

def Plv(x1, Emax, Emin, t, Tc):
    return Elastance(Emax, Emin, t, Tc) * x1

def Elastance(Emax, Emin, t, Tc):
    t = t - int(t/Tc) * Tc
    tn = t / (0.2 + 0.15 * Tc)
    return (Emax - Emin) * 1.55 * (tn / 0.7)**1.9 / ((tn / 0.7)**1.9 + 1) * 1 / ((tn / 1.17)**21.9 + 1) + Emin

def pvloop_simulator_full(Tc, start_v, Emax, Emin, Rm, Ra, Vd, N, plotloops, plotpressures, plotflow):
    startp = 75.
    Rs = 1.0
    Rc = 0.0398
    Ca = 0.08
    Cs = 1.33
    Cr = 4.400
    Ls = 0.0005

    start_pla = float(start_v * Elastance(Emax, Emin, 0, Tc))
    start_pao = startp
    start_pa = start_pao
    start_qt = 0
    y0 = [start_v, start_pla, start_pa, start_pao, start_qt]

    t = np.linspace(0, Tc * N, int(60000 * N))
    sol = odeint_standard(heart_ode, y0, t, args=(Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, Emax, Emin, Tc))

    result_Vlv = np.array(sol[:, 0]) + Vd
    result_Plv = np.array([Plv(v, Emax, Emin, xi, Tc) for xi, v in zip(t, sol[:, 0])])

    ved = sol[(N-1) * 60000, 0] + Vd
    ves = sol[200 * int(60/Tc) + 9000 + (N-1) * 60000, 0] + Vd
    ef = (ved - ves) / ved * 100.

    X = [result_Vlv[(N-1) * 60000 + 12000], result_Vlv[(N-1) * 60000 + 24000], result_Vlv[(N-1) * 60000 + 36000], result_Vlv[(N-1) * 60000 + 48000]]

    minv = min(result_Vlv[(N-1) * 60000 : N * 60000 - 1])
    minp = min(result_Plv[(N-1) * 60000 : N * 60000 - 1])
    maxp = max(result_Plv[(N-1) * 60000 : N * 60000 - 1])

    ved2 = sol[(N-1) * 60000 - 1, 0] + Vd
    isperiodic = 0
    if abs(ved - ved2) > 5.:
        isperiodic = 1

    if plotloops:
        plt.plot(result_Vlv[(N-2) * 60000 : N * 60000], result_Plv[(N-2) * 60000 : N * 60000])
        plt.xlabel("LV volume (ml)")
        plt.ylabel("LV pressure (mmHg)")
        plt.show()

    if plotpressures:
        result_Pla = np.array(sol[:, 1])
        result_Pa = np.array(sol[:, 2])
        result_Pao = np.array(sol[:, 3])
        plt.plot(t[(N-2) * 60000 : N * 60000], result_Plv[(N-2) * 60000 : N * 60000], label='LV P')
        plt.plot(t[(N-2) * 60000 : N * 60000], result_Pao[(N-2) * 60000 : N * 60000], label='Aortic P')
        plt.plot(t[(N-2) * 60000 : N * 60000], result_Pa[(N-2) * 60000 : N * 60000], label='Arterial P')
        plt.plot(t[(N-2) * 60000 : N * 60000], result_Pla[(N-2) * 60000 : N * 60000], label='Left atrial P')
        plt.xlabel("Time (s)")
        plt.ylabel("Pressure (mmHg)")
        plt.legend(loc='upper right', framealpha=1)
        plt.show()

    if plotflow:
        result_Q = np.array(sol[:, 4])
        plt.plot(t[(N-2) * 60000 : N * 60000], result_Q[(N-2) * 60000 : N * 60000])
        plt.xlabel("Time (s)")
        plt.ylabel("Blood flow (ml/s)")
    X = sol[(N-1) * 60000 : N * 60000, ]
    ted = t[(N-1) * 60000]
    tes = t[200 * int(60 / Tc) + 9000 + (N-1) * 60000]
    t = t[(N-1) * 60000 : N * 60000]

    return X, t, ved, ves, ef, ted, tes

def get_batch(n, device, Tc, start_v, Emax, Emin, Rm, Ra, Vd):
    X, t, ved, ves, ef, ted, tes = pvloop_simulator_full(Tc, start_v, Emax, Emin, Rm, Ra, Vd, 70, True, False, False)
    t = t - t[0]
    ted = ted - t[0]
    tes = tes - t[0]
    indices = np.linspace(0, len(t) - 1, num=n, dtype=int)
    X_tensor = torch.tensor(X[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    t_tensor = torch.tensor(t[indices], dtype=torch.float32).to(device)
    X_batch = torch.tensor(X[indices], dtype=torch.float32).unsqueeze(1).unsqueeze(1).to(device)
    return X_tensor, t_tensor, X_batch, ved, ves, ted, tes

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 50),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 5),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



def main(args):
    intervals = [[0.4, 1.7],   # Tc
                 [0., 280.],   # start_v
                 [0.5, 3.5],   # Emax
                 [0.02, 0.1],  # Emin
                 [0.005, 0.1], # Rm
                 [0.0001, 0.25], # Ra
                 [4, 25]]      # Vd

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    start_total = time.time()

    for loop in range(1, args.loops + 1):  # Run training loop `args.loops` times
        print(f"#####{loop} training #####")
        
        Tc = np.random.uniform(*intervals[0])
        start_v = np.random.uniform(*intervals[1])
        Emax = np.random.uniform(*intervals[2])
        Emin = np.random.uniform(*intervals[3])
        Rm = np.random.uniform(*intervals[4])
        Ra = np.random.uniform(*intervals[5])
        Vd = np.random.uniform(*intervals[6])
        
        print(f"Tc: {Tc:.4f}, start_v: {start_v:.4f}, Emax: {Emax:.4f}, Emin: {Emin:.4f}, Rm: {Rm:.4f}, Ra: {Ra:.4f}, Vd: {Vd:.4f}")
        
        batch_y0, batch_t, batch_y, batch_ved, batch_ves, batch_ted, batch_tes = get_batch(60, device, Tc, start_v, Emax, Emin, Rm, Ra, Vd)
        t = batch_t

        func = ODEFunc().to(device)
        optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
        end = time.time()

        time_meter = RunningAverageMeter(0.97)
        loss_meter = RunningAverageMeter(0.97)

        for itr in range(1, args.iterations + 1):  # Number of iterations from `args.iterations`
            optimizer.zero_grad()
            pred_y = odeint(func, batch_y0, batch_t).to(device)
            loss = torch.mean(torch.abs(pred_y - batch_y))
            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - end)
            loss_meter.update(loss.item())

            if itr % 50 == 0:
                print(f'Training Loss: {loss.item()}')
                with torch.no_grad():
                    pred_y = odeint(func, batch_y0, torch.tensor([batch_ted, batch_tes]))
                    pred_vol_EF = pred_y.squeeze().cpu().numpy()[:, 0]
                    print(f'EF pred: {(pred_vol_EF[0] - pred_vol_EF[1]) / pred_vol_EF[0]}')
                    print(f'EF true: {(batch_ved - batch_ves) / batch_ved}')
                    loss_EF = np.mean(np.abs((pred_vol_EF[0] - pred_vol_EF[1]) / pred_vol_EF[0] - (batch_ved - batch_ves) / batch_ved))
                    print(f'Iter {itr:04d} | EF Loss {loss_EF:.6f}')
                    pred_y = odeint(func, batch_y0, batch_t).to(device)

            end = time.time()

    end_total = time.time()
    print("Total time:", end_total - start_total)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural ODE for Cardiac Model')
    
    parser.add_argument('--loops', type=int, default=10, help='Number of training loops')
    parser.add_argument('--iterations', type=int, default=2000, help='Number of iterations per loop')

    args = parser.parse_args()

    main(args)