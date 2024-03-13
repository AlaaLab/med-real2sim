import matplotlib.pylab as plt
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import os
from skimage.transform import rescale, resize
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
from scipy.interpolate import interp1d

#for pvloop simulator:
import pandas as pd
from scipy.integrate import odeint 
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot
import sys
import numpy as np

#odesolver:
from torch.storage import T
import argparse
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description='pssl with 7 param training code')
    parser.add_argument('--output_path', type=str, default='', help='Output path for saving files')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--pretext_model_path', type=str, default='', help='Input path for pretext model weight')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    # parser.add_argument('--val_size', type=int, default=50, help='Size of validation set')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--ID', type=str, default='CAMUS_7param_Vloss', help='Identifier for the experiment')
    parser.add_argument('--camus_input_directory', type=str, default='', help='Input directory for CAMUS dataset')
    args = parser.parse_args()
    return args


class CardioDataset(torch.utils.data.Dataset):
    def __init__(self, patient_ids, sequences, edvs, esvs, efs):
        self.patient_ids = patient_ids
        self.sequences = sequences
        self.edvs = edvs
        self.esvs = esvs
        self.efs = efs

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        sequence = self.sequences[idx]
        edv = self.edvs[idx]
        esv = self.esvs[idx]
        ef = self.efs[idx]
        
        data = {'patient':patient_id,
                '4CH_sequence': sequence,
                'EDV': edv,
                'ESV': esv,
                'EF': ef   
        }
        return data

class IntervalNormalizationLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # new_output = [Tc, start_p, Emax, Emin, Rm, Ra, Vd]
        self.a = torch.tensor([0.4, 0., 0.5, 0.02, 0.005, 0.0001, 4.], dtype=torch.float32) #HR in 20-200->Tc in [0.3, 4]
        self.b = torch.tensor([1.7, 280., 3.5, 0.1, 0.1, 0.25, 24.], dtype=torch.float32)
        #taken out (initial conditions): a: 20, 5, 50; b: 400, 20, 100
    def forward(self, inputs):
        sigmoid_output = torch.sigmoid(inputs)
        scaled_output = sigmoid_output * (self.b - self.a) + self.a
        return scaled_output

class NEW3DCNN(nn.Module): 
    def __init__(self, num_parameters): 
        super(NEW3DCNN, self).__init__() 
        
        self.conv1      = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(8)
        self.conv2      = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm3d(16)
        self.conv3      = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm3d(32)
        self.conv4      = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm3d(64)
        self.conv5      = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm3d(128)
        self.pool       = nn.AdaptiveAvgPool3d(1)
        self.fc1        = nn.Linear(128, 512)
        self.fc2        = nn.Linear(512, num_parameters)
        self.norm1      = IntervalNormalizationLayer()
        
    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = F.relu(self.batchnorm5(self.conv5(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm1(x)

        
        return x


# Define a neural network with one hidden layer
class Interpolator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 250).double()
        self.fc2 = nn.Linear(250, 2).double()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Accessing parsed arguments
    output_path = args.output_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    pretext_model_path = args.pretext_model_path
    ID = args.ID
    camus_input_directory = args.echonet_input_directory

    path = camus_input_directory
    file = f"{ID}_epoch_{num_epochs}_lr_{learning_rate}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sequences_all = []
    info_data_all = []

    for i in range(1, 6):
        sequences_batch = np.load(f'{path}/sequences_{i}.npy', allow_pickle=True)
        info_data_batch = pd.read_csv(f'{path}/info_data_{i}.csv')
        
        sequences_all.append(sequences_batch)
        info_data_all.append(info_data_batch)
        
    # Concatenate the sequence arrays into a single array
    sequences_all = np.concatenate(sequences_all, axis=0)

    # Concatenate the info_data DataFrames into a single DataFrame
    info_data_all = pd.concat(info_data_all, axis=0, ignore_index=True)

    data_size = 450
    patient_id = torch.tensor(range(data_size), device=device)
    LVedv = torch.tensor(info_data_all['LVedv'].values, device=device)
    LVesv = torch.tensor(info_data_all['LVesv'].values, device=device)
    LVef = torch.tensor(info_data_all['LVef'].values, device=device)
    sequences_all = torch.tensor(sequences_all, device=device)


    train_data = CardioDataset(patient_ids=patient_id, sequences = sequences_all, edvs = LVedv, esvs= LVesv, efs = LVef)

    print("Done loading training data!")
    # define normalization layer to make sure output xi in an interval [ai, bi]:
    # define normalization layer to make sure output xi in an interval [ai, bi]:


    # Initialize the neural network
    net = Interpolator()
    net.load_state_dict(torch.load(pretext_model_path))
    print("Done loading the pretext model!")

    model = NEW3DCNN(num_parameters = 7)
    model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create the data loader

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)



    # loading validation set
    sequences_all = []
    info_data_all = []
    sequences_batch = np.load(f'{path}/test/sequences_test.npy', allow_pickle=True)
    info_data_batch = pd.read_csv(f'{path}/test/info_data_test.csv')

    sequences_all.append(sequences_batch)
    info_data_all.append(info_data_batch)
        
    # Concatenate the sequence arrays into a single array
    sequences_all = np.concatenate(sequences_all, axis=0)

    # Concatenate the info_data DataFrames into a single DataFrame
    info_data_all = pd.concat(info_data_all, axis=0, ignore_index=True)

    patient_id = torch.tensor(range(50), device=device)
    LVedv = torch.tensor(info_data_all['LVedv'].values, device=device)
    LVesv = torch.tensor(info_data_all['LVesv'].values, device=device)
    LVef = torch.tensor(info_data_all['LVef'].values, device=device)
    sequences_all = torch.tensor(sequences_all, device=device)

    testing_data = CardioDataset(patient_ids=patient_id, sequences = sequences_all, edvs = LVedv, esvs= LVesv, efs = LVef)
    test_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

    test_data = next(iter(test_loader))
    test_seq = test_data['4CH_sequence']
    test_seq = test_seq.reshape(batch_size, 1, 30, 112,112)
    test_tensor = torch.tensor(test_seq, dtype=torch.float32) 

    test_EF = test_data['EF']

    print("Done loading validation set!")



    # Training
    num_epochs = num_epochs
    best_mae = float('inf')
    patience = 7
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=patience, verbose=True)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for j, batch in enumerate(train_loader):
            optimizer.zero_grad()
            seq = batch["4CH_sequence"]
            seq = seq.reshape(batch_size, 1, 30, 112,112)
            input_tensor = torch.tensor(seq, dtype=torch.float32) 

            #simulated values: sim_output = (V_ED, V_ES)
            x = model(input_tensor).double()
            x1 = x[:, :6]
            Vd = x[:, -1:]
            output1 = net(x1)
            output = output1 + Vd-4
            ved, ves= torch.split(output, split_size_or_sections=1, dim=1)
            ef = (ved-ves)/ved*100

            trueV_ED = torch.tensor(batch['EDV'])
            trueV_ES = torch.tensor(batch['ESV'])
            true_EF = torch.tensor(batch['EF'])
            #criterion = torch.nn.MSELoss()
            loss = criterion(ved.flatten(), trueV_ED) + criterion(ves.flatten(), trueV_ES)
            #loss = criterion(ved, trueV_ED) + criterion(ves, trueV_ES) + criterion(ef, true_EF)
            #loss = criterion(ef.flatten(), true_EF)
            # Compute the gradients and update the model parameters using backpropagation
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        with torch.no_grad():
            test_x = model(test_tensor).double()
            test_x1 = test_x[:, :6]
            test_Vd = test_x[:, -1:]
            test_output1 = net(test_x1)
            test_output = test_output1 + test_Vd -4
            a, b = torch.split(test_output, split_size_or_sections=1, dim=1)
            #test_sim_EF = (a-b)/a*100
            test_sim_EF = (a-b)/a*100
        test_EF_np = test_EF.numpy()
        test_sim_EF_np = test_sim_EF.detach().numpy()
        MAE = np.mean(np.abs(test_EF_np - test_sim_EF_np.flatten()))
        #lr_scheduler.step(MAE)
        print("Epoch [{}/{}], Loss: {:.4f}, valid MAE: {:.4f}".format(epoch+1, num_epochs, epoch_loss, MAE))
        if MAE < best_mae:
            best_mae = MAE
            file_label = f"epoch:{epoch+1} MAE:{MAE:.2f}"
            print(file_label)
            test_x = model(test_tensor).double()
            test_x1 = test_x[:, :6]
            test_Vd = test_x[:, -1:]
            test_output1 = net(test_x1)
            test_output = test_output1 + test_Vd -4
            a, b = torch.split(test_output, split_size_or_sections=1, dim=1)
            test_sim_EF = (a-b)/a*100
            test_EF_np = test_EF.numpy()
            test_sim_EF_np = test_sim_EF.detach().numpy()

            combined = torch.cat((test_sim_EF, test_EF.unsqueeze(1), test_x), dim=1)
            np_array = combined.detach().numpy()
            np.savetxt(f'{output_path}/{file}_best_model.csv', np_array, delimiter=',', header='sim_EF,trueEF, Tc, start_v, Emax, Emin, Rm, Ra, Vd')
            torch.save(model.state_dict(), f'{output_path}/{file}_weight_best_model.pt')
        

    torch.save(model.state_dict(), os.path.join(output_path,f'{file}_end_weight.py'))


if __name__ == "__main__":
    main()