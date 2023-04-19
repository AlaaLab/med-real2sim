#for the NN interpolator (net):


# Define a neural network with one hidden layer
class Interpolator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64).double()
        self.fc2 = nn.Linear(64, 1).double()

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = self.fc2(z)
        return z

# Initialize the neural network
net = Interpolator()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

#for the inverse NN (invnet):

class IntervalNormalizationLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # new_output = [start_v, start_paos, emax, emin, rc, rs, cs]
        # output = (Tc =1.0, Rs, Rm = 0.005, Ra = 0.001, Rc, Ca = 0.08, Cs, Cr = 4.44, Ls = 0.0005, start_v, start_pla, start_pao). start_v: 20-400; start_pla: 5-20, start_pao: 50-100.
        self.a = torch.tensor([0.5, 15., 0.2], dtype=torch.float32) #HR in 20-200->Tc in [0.3, 4]
        self.b = torch.tensor([2.0, 400., 30.], dtype=torch.float32)
        #taken out (initial conditions): a: 20, 5, 50; b: 400, 20, 100
    def forward(self, inputs):
        sigmoid_output = torch.sigmoid(inputs)
        scaled_output = sigmoid_output * (self.b - self.a) + self.a
        return scaled_output

class INVNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 64).double()
        self.fc2 = nn.Linear(64, 3).double()
        self.norm1 = IntervalNormalizationLayer()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.norm1(self.fc2(x))

        return x

# Initialize the neural network
invnet = INVNN()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(invnet.parameters(), lr=0.000001)
