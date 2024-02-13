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
import os
import itertools
from math import prod

# Specify the file paths
output_path = '/accounts/biost/grad/keying_kuang/ML/interpolator_RaRm_10'
points_file = 'points_reverse.pt'
vs_file = 'Vs_reverse.pt'

# Load the data points
points2 = torch.load(os.path.join(output_path, points_file))
equally_spaced_points_tensor = torch.load(os.path.join(output_path, vs_file))

# Convert to the desired data types if needed
x = torch.tensor(points2, dtype=torch.float64)
y = torch.tensor(equally_spaced_points_tensor, dtype=torch.float64)

n_pars = len(points2[0])
n_neurons = 256 ##in general 256 is enough
lr = 0.01
threshold_train = 1.
threshold_test = 2.
n_epochs = 30000

class Interpolator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_pars, n_neurons).double()
        self.fc2 = nn.Linear(n_neurons, 10).double()
    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = self.fc2(z)
        return z

# Initialize the neural network
net = Interpolator()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
losses = []
i = 0

while True:
  for epoch in range(n_epochs):
      # Forward pass
      y_pred = net(x)
      loss = criterion(y_pred, y)

      # Backward pass and optimization
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Print progress
      if epoch % 5000 == 0:
          print(f'Epoch {epoch}, loss: {loss.item():.4f}')
          losses.append(loss.item())

  if abs(losses[-1] - losses[-2])<5.:
    lr = lr / 10.
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
  if (i>10 and losses>50.):
    n_neurons += 24
    net = Interpolator()
  if (i>30): break
  if (losses[-1]<threshold_train): break
  i += 1

print("Training error", loss.item())

torch.save(net.state_dict(), '/accounts/biost/grad/keying_kuang/ML/interpolator_RaRm_10/interpRaRm_param_weight_2_EStoED.pt')


