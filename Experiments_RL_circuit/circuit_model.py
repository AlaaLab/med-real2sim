# this file is to generate voltage V from current I and static features [R,L] using neuroODE
import utils as utils
import numpy as np
import os
import argparse
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchdiffeq import odeint_adjoint as odeint

data_size = 4000
batch_time = 50
batch_size = 20
niters = 2000
test_freq = 20
device = 'cpu'
viz = True

### generate voltage V one by one using neural-ODE
n_obs = 1
#t = range(data_size)
#t = torch.tensor(t, dtype=torch.float32)
t = torch.linspace(0.,20.,data_size)

ecg_data = utils.simulate_ecg_data(n_patients= n_obs)
ecg_data_np = np.array(ecg_data).reshape(n_obs, 4000)

voltages = utils.generate_voltage(ecg_data)
voltages_np = np.array(voltages).reshape(n_obs, 4000)

signal_rates = utils.get_signal_rate(ecg_data)
hrv = np.array(signal_rates).reshape(n_obs,80)
hrv = pd.DataFrame(hrv)

# the first column is the means of the RR intervals (signal rates)
# the second column is the standard deviations of the RR intervals
# so we will use these two columns as our true static features 
static_features = hrv.iloc[:,[0,1]]
static_features.columns = ["RR_mean", "RR_std"]
R = static_features.iloc[0,0]
L = static_features.iloc[0,1]


### for each patient, solve for V(t) using neuroODE

# initial value of the current
true_I0 = torch.tensor([1.])
true_I0 = nn.Parameter(true_I0)

# transpose currents_np to get currents_new
voltages_new = voltages_np.T
# transform currents_new to torch tensor
voltages = torch.from_numpy(voltages_new)
true_voltages = torch.stack([voltages, voltages], dim = -1)
true_voltages = true_voltages.to(dtype=torch.float32)
#plt.figure()
#plt.plot(t.numpy(), voltages.numpy()[:,0])
#plt.show()

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False)) # some random start points
    batch_v0 = true_voltages[s]  
    batch_t = t[:batch_time]  
    batch_v = torch.stack([true_voltages[s + i] for i in range(batch_time)], dim=0)  # (T, M, D) get true values over T
    return batch_v0.to(device), batch_t.to(device), batch_v.to(device)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

#### Visualization 
if viz:
    makedirs("Circuit Model")
    import matplotlib.pyplot as plt

def visualize(true_y, pred_y, odefunc, itr):
  
    if viz:
      
        plt.figure()
        plt.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        plt.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        plt.savefig(  '/ts' + str(itr) + '.png')
        plt.show()
        
        plt.figure()
        plt.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        plt.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        plt.savefig('/phase' + str(itr) + '.png')
        plt.show()

#### Neural ODE Model
class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 150),
            nn.ReLU(150),
            nn.Tanh(),
            nn.ReLU(150),
            nn.Linear(150, 1),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

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


ii = 0

func = ODEFunc().to(device)
    
optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
end = time.time()

time_meter = RunningAverageMeter(0.97)
    
loss_meter = RunningAverageMeter(0.97)

for itr in range(1, niters + 1):
    optimizer.zero_grad()
    pred_y = odeint(func, true_I0, t).to(device)
    pred_y_deriv = func(t, pred_y)
    voltages_pred = pred_y*R + L*pred_y_deriv
    loss = torch.mean(torch.abs(voltages - voltages_pred))
    loss.backward()
    optimizer.step()

    time_meter.update(time.time() - end)
    loss_meter.update(loss.item())

    if itr % test_freq == 0:
        with torch.no_grad():
            pred_y = odeint(func, true_I0, t).to(device)
            pred_y_deriv = func(t, pred_y)
            loss = torch.mean(voltages - voltages_pred)
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
            #visualize(true_y, pred_y, func, ii)
            ii += 1

    end = time.time()


    ##### after getting the function of dI/dt, we can use it to get time series data for dI(t)/dt #####
    ##### we can use the time series data for I(t) and for dI(t)/dtto get the time series data for V(t) #####

