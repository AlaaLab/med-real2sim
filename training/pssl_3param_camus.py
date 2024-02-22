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
    parser = argparse.ArgumentParser(description='pssl with 3 param training code')
    parser.add_argument('--output_path', type=str, default='', help='Output path for saving files')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training')
    parser.add_argument('--interpolator_path', type=str, default='', help='Input path for Interpolator weight')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    # parser.add_argument('--val_size', type=int, default=50, help='Size of validation set')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--ID', type=str, default='CAMUS_3param_Vloss', help='Identifier for the experiment')
    parser.add_argument('--camus_input_directory', type=str, default='', help='Input directory for CAMUS dataset')
    args = parser.parse_args()
    return args

class IntervalNormalizationLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # new_output = [Tc, start_v, Emax]
        self.a = torch.tensor([0.5, 15, 0.3], dtype=torch.float32) #HR in 20-200->Tc in [0.3, 4]
        self.b = torch.tensor([2, 400, 30], dtype=torch.float32)
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
        self.fc1 = nn.Linear(3, 64).double()
        self.fc2 = nn.Linear(64, 2).double()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
                '2CH_sequence': sequence,
                'EDV': edv,
                'ESV': esv,
                'EF': ef   
        }
        return data
def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Accessing parsed arguments
    output_path = args.output_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    interpolator_path = args.interpolator_path
    ID = args.ID
    camus_input_directory = args.echonet_input_directory

    path = camus_input_directory
    file = f"{ID}_epoch_{num_epochs}_lr_{learning_rate}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sequences_all = []
    info_data_all = []


    for i in range(1, 6):
        sequences_batch = np.load(f'{path}/sequences_{i}.npy', allow_pickle=True)
        info_data_batch = pd.read_csv(f'{path}/info/info_data_{i}.csv')
        
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

    # Initialize the neural network
    net = Interpolator()
    net.load_state_dict(torch.load(interpolator_path))
    print("Done loading the interpolator3")

    model = NEW3DCNN(num_parameters = 3)
    model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create the data loader

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)



    # loading validation set
    sequences_all = []
    info_data_all = []
    sequences_batch = np.load(f'{path}/input/sequences_test.npy', allow_pickle=True)
    info_data_batch = pd.read_csv(f'{path}/input/info_data_test.csv')

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
    test_seq = test_data['2CH_sequence']
    test_seq = test_seq.reshape(batch_size, 1, 30, 256, 256)
    test_tensor = torch.tensor(test_seq, dtype=torch.float32) 

    test_EF = test_data['EF']

    print("Done loading validation set!")

    # Training
    num_epochs = num_epochs
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for j, batch in enumerate(train_loader):
            optimizer.zero_grad()
            seq = batch["2CH_sequence"]
            seq = seq.reshape(batch_size, 1, 30, 256, 256)
            input_tensor = torch.tensor(seq, dtype=torch.float32) 

            #simulated values: sim_output = (V_ED, V_ES)
            x = model(input_tensor).double()
            output = net(x)
            ved, ves= torch.split(output, split_size_or_sections=1, dim=1)
            ef = (ved-ves)/ved*100

            trueV_ED = torch.tensor(batch['EDV'])
            trueV_ES = torch.tensor(batch['ESV'])
            true_EF = torch.tensor(batch['EF'])
            #criterion = torch.nn.MSELoss()
            #loss = criterion(ved.flatten(), trueV_ED) + criterion(ves.flatten(), trueV_ES)
            #loss = criterion(ved, trueV_ED) + criterion(ves, trueV_ES) + criterion(ef, true_EF)
            loss = criterion(ef.flatten(), true_EF)

            # Compute the gradients and update the model parameters using backpropagation
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        with torch.no_grad():
            test_x = model(test_tensor).double()
            test_output = net(test_x)
            a, b = torch.split(output, split_size_or_sections=1, dim=1)
            test_sim_EF = (a-b)/a*100
        test_EF_np = test_EF.numpy()
        test_sim_EF_np = test_sim_EF.detach().numpy()
        MAE = np.mean(np.abs(test_EF_np - test_sim_EF_np.flatten()))
        print("Epoch [{}/{}], Loss: {:.4f}, valid MAE: {:.4f}".format(epoch+1, num_epochs, epoch_loss, MAE))
        if (epoch+1) in [50, 100, 150, 200, 250, 300] or MAE < 10:
            file_label = f"epoch_{epoch}_MAE_{MAE:.2f}"
            combined = torch.cat((test_sim_EF, test_EF.unsqueeze(1), test_x), dim=1)
            np_array = combined.detach().numpy()
            np.savetxt(f'{output_path}/{file}_validresult_{file_label}.csv', np_array, delimiter=',', header='sim_EF,trueEF, a, v, c')
            torch.save(model.state_dict(), f'{output_path}/{file}_validweight_{file_label}.pt')
        

    torch.save(model.state_dict(), os.path.join(output_path,f'{file}_end_weight.py'))

    ########## testing data #####################
    sequences_all = []
    info_data_all = []

    sequences_batch = np.load(f'{path}/input/sequences_test.npy', allow_pickle=True)
    info_data_batch = pd.read_csv(f'{path}/input/info_data_test.csv')

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
    test_loader = DataLoader(testing_data, batch_size=50, shuffle=True)

    test_data = next(iter(test_loader))
    test_seq = test_data['2CH_sequence']
    test_seq = test_seq.reshape(50, 1, 30, 256, 256)
    test_tensor = torch.tensor(test_seq, dtype=torch.float32) 


    x = model(test_tensor).double()
    output = net(x)
    ved, ves = torch.split(output, split_size_or_sections=1, dim=1)
    ef = (ved-ves)/ved*100
    # initialize empty lists for sim_EF and trueEF
    sim_EF = ef
    true_EF = test_data['EF']


    combined = torch.cat((sim_EF, true_EF.unsqueeze(1), x), dim=1)

    np_array = combined.detach().numpy()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, f'{file}_testresult.csv')  # set the output file path

    # save the data to the output file
    np.savetxt(output_file, np_array, delimiter=',', header='sim_EF,trueEF, Tc, start_v, start_pao, Rc, Emax, Emin, Vd')

    ###### training data ############
    train_data = next(iter(train_loader))
    train_seq = train_data['2CH_sequence']
    train_seq = train_seq.reshape(batch_size, 1, 30, 256, 256)
    train_tensor = torch.tensor(train_seq, dtype=torch.float32) 


    x = model(train_tensor).double()
    output = net(x)
    ved, ves = torch.split(output, split_size_or_sections=1, dim=1)
    ef = (ved-ves)/ved*100
    # initialize empty lists for sim_EF and trueEF
    sim_EF = ef
    true_EF = train_data['EF']


    combined = torch.cat((sim_EF, true_EF.unsqueeze(1), x), dim=1)

    np_array = combined.detach().numpy()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = os.path.join(output_path, f'{file}_trainresult.csv')  # set the output file path

    # save the data to the output file
    np.savetxt(output_file, np_array, delimiter=',', header='sim_EF,trueEF, Tc, start_v, start_pao, Rc, Emax, Emin, Vd')



if __name__ == "__main__":
    main()