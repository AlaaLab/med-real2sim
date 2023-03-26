# -*- coding: utf-8 -*-
"""main model (3DCNN).ipynb
# main model (3DCNN)
"""

'''
Estimates the 9 parameters of the model: Tc, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls.
The initial conditions (start_v, start_pla, start_pao) are assumed constant, defined as true_y0 (and assumed start_pa=start_pao and start_Qt=0).
Emin=2.00 and Emax=0.06 are assumed. Vd has no effect on the equations so cannot be estimated.
'''

from google.colab import drive
drive.mount('/content/drive')

#!pip install SimpleITK
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import os
from skimage.transform import rescale, resize
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np

#for norm:
import tensorflow as tf

#new:
from torch.storage import T
import argparse
import time

parser = argparse.ArgumentParser('ODE demo')

#parameters for the simulation:
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=500) #size of true_y and pred_y and t (n of timepoints for the diff eq in odeint)
#parser.add_argument('--batch_time', type=int, default=100) #batch_t takes the first batch_time(=10) first elements of t
parser.add_argument('--int_time', type=float, default=2.) #added
#parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args(args=[])

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[140., 8.2, 77, 77, 0]]).to(device) #true_y0 = [Vlv-Vd, Pla, Pa, Pao, Qt] at t=0 (in ED)
t = torch.linspace(0., args.int_time, args.data_size).to(device)


def display_video(video):
#   ref: https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python 
    plt.figure(figsize=(20,16))
    plt.gray()
    plt.subplots_adjust(0,0,1,1,0.01,0.01)
    for i in range(video.shape[0]):
        plt.subplot(5,6,i+1), plt.imshow(video[i]), plt.axis('off')
        # plt.savefig("image.jpg"))
    plt.show()

class CamusIterator(Dataset):
    def __init__( 
        self, 
        data_type='train', 
        global_transforms=[], 
        augment_transforms=[]
    ):
        super(CamusIterator, self).__init__()
        
        train_file='/content/drive/MyDrive/in-silico-ML/CAMUS_public/database/training'
        test_file='/content/drive/MyDrive/in-silico-ML/CAMUS_public/database/testing'
        
        if data_type == 'train':
            data_file = train_file
        elif data_type == 'test':
            data_file = test_file
        else:
            raise Exception('Wrong data_type for CamusIterator')
            
        self.data_type = data_type
        self.data_file = data_file
        self.global_transforms = global_transforms
        self.augment_transforms = augment_transforms
    
    def __read_image( self, patient_file, suffix ):
        image_file = '{}/{}/{}'.format( self.data_file, patient_file, patient_file+suffix )
        # https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python
        image = sitk.GetArrayFromImage( sitk.ReadImage(image_file, sitk.sitkFloat32) )
        return image

    def __read_info( self, data_file ):
        info = {}
        with open( data_file, 'r' ) as f:
            for line in f.readlines():
                info_type, info_details = line.strip( '\n' ).split( ': ' )
                info[ info_type ] = info_details
        return info

    def __len__( self ):
        return len( os.listdir(self.data_file) )
    
    def __getitem__( self, index ):
        patient_file = 'patient{}'.format( f'{index+1:04}' ) # patient{0001}, patient{0002}, etc
        
        image_2CH_ED = self.__read_image( patient_file, '_2CH_ED.mhd' )
        image_2CH_ES = self.__read_image( patient_file, '_2CH_ES.mhd' )
        image_4CH_ED = self.__read_image( patient_file, '_4CH_ED.mhd' )
        image_4CH_ES = self.__read_image( patient_file, '_4CH_ES.mhd' )
        image_2CH_sequence = self.__read_image( patient_file, '_2CH_sequence.mhd' )
        image_4CH_sequence = self.__read_image( patient_file, '_4CH_sequence.mhd' )
        
        if self.data_type == 'train':
            image_2CH_ED_gt = self.__read_image( patient_file, '_2CH_ED_gt.mhd' )
            image_2CH_ES_gt = self.__read_image( patient_file, '_2CH_ES_gt.mhd' )
            image_4CH_ED_gt = self.__read_image( patient_file, '_4CH_ED_gt.mhd' )
            image_4CH_ES_gt = self.__read_image( patient_file, '_4CH_ES_gt.mhd' )

        info_2CH = self.__read_info( '{}/{}/{}'.format(self.data_file, patient_file, 'Info_2CH.cfg') )
        info_4CH = self.__read_info( '{}/{}/{}'.format(self.data_file, patient_file, 'Info_4CH.cfg') )
        
        if self.data_type == 'train':
            data = {
                'patient': patient_file,
                '2CH_ED': image_2CH_ED,
                '2CH_ES': image_2CH_ES,
                '4CH_ED': image_4CH_ED,
                '4CH_ES': image_4CH_ES,
                '2CH_sequence': image_2CH_sequence,
                '4CH_sequence': image_4CH_sequence,
                '2CH_ED_gt': image_2CH_ED_gt,
                '2CH_ES_gt': image_2CH_ES_gt,
                '4CH_ED_gt': image_4CH_ED_gt,
                '4CH_ES_gt': image_4CH_ES_gt,
                'info_2CH': info_2CH,    # Dictionary of infos
                'info_4CH': info_4CH}    # Dictionary of infos
        elif self.data_type == 'test':
            data = {
                'patient': patient_file,
                '2CH_ED': image_2CH_ED,
                '2CH_ES': image_2CH_ES,
                '4CH_ED': image_4CH_ED,
                '4CH_ES': image_4CH_ES,
                '2CH_sequence': image_2CH_sequence,
                '4CH_sequence': image_4CH_sequence,
                'info_2CH': info_2CH,   # Dictionary of infos
                'info_4CH': info_4CH}   # Dictionary of infos
        
        # Transforms
        for transform in self.global_transforms:
            data = transform(data)
        for transform in self.augment_transforms:
            data = transform(data)
            
        return data

    def __iter__( self ):
        for i in range( len(self) ):
            yield self[ i ]

param_Loader = {'batch_size': 1,
                'shuffle': True,
                'num_workers': 8}

class ResizeImagesAndLabels(object):
    ''' 
    Ripped out of Prof. Stough's code 
    '''
    
    def __init__(self, size, fields=['2CH_ED', '2CH_ES', '4CH_ED', '4CH_ES',
                                     '2CH_ED_gt', '2CH_ES_gt', '4CH_ED_gt', '4CH_ES_gt']):
        self.size = size
        self.fields = fields
        
    def __call__(self, data):
        for field in self.fields:            
            # transpose to go from chan x h x w to h x w x chan and back.
            data[field] = resize(data[field].transpose([1,2,0]), 
                                 self.size, mode='constant', 
                                 anti_aliasing=True)
            data[field] = data[field].transpose( [2,0,1] )      

        return data

global_transforms = [
    ResizeImagesAndLabels(size=[256, 256])
]
augment_transforms = [
    #AddSaltPepper(freq = .1)
]


train_data = CamusIterator(
    data_type='train',
    global_transforms=global_transforms,
    augment_transforms=augment_transforms,
)

import numpy as np
from skimage.transform import resize

def padding(sequence, n, add, size=(256,256)): #make the video of n frames (chosen at n=30 below)
    n_frames = sequence.shape[0+add]
    height = sequence.shape[1+add]
    width = sequence.shape[2+add]
    ratio = height / width

    # Resize each frame to have the same size.
    new_frames = []
    for i in range(n_frames):
        frame = sequence[0][i]
        new_frame = resize(frame, size, mode='constant', anti_aliasing=True) #resize to (256,256)
        new_frames.append(new_frame)
    new_sequence = np.stack(new_frames, axis=0)

    # Pad or truncate the sequence to have 30 frames.
    if n_frames < n:
        #padding = np.zeros((n - n_frames, size[0], size[1])) #create n - n_frames imgs of 0's of size (256,256)
        #new_sequence = np.concatenate([new_sequence, padding], axis=0)

        padding = np.zeros((n - n_frames, size[0], size[1]))
        new_sequence = np.concatenate([new_sequence, padding], axis=0)

    elif n_frames > n:
        new_sequence = new_sequence[:n] #only keep the first n images (=10)

    return new_sequence

import torch.nn.functional as F

#for having each value of the output xi in an interval [ai, bi]:
class IntervalNormalizationLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #output = (Tc, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls, start_v, start_pla, start_pao). start_v: 20-400; start_pla: 5-20, start_pao: 50-100.
        self.a = torch.tensor([0.5, 0.5, 0.002, 0.01, 0.01, 0.05, 1.0, 3., 0.00001], dtype=torch.float32) #HR in 20-200->Tc in [0.3, 4]
        self.b = torch.tensor([1.5, 1.5, 0.008, 0.10, 0.06, 1.80, 1.6, 6., 0.0015], dtype=torch.float32)
        #taken out (initial conditions): a: 20, 5, 50; b: 400, 20, 100
    def forward(self, inputs):
        sigmoid_output = torch.sigmoid(inputs)
        scaled_output = sigmoid_output * (self.b - self.a) + self.a
        return scaled_output

class NEW3DCNN(nn.Module):
    def __init__(self, num_parameters):
        super(NEW3DCNN, self).__init__()

        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1) #from 1 channel to 8, keep same size
        self.batchnorm1 = nn.BatchNorm3d(8) #batch normalization on img with 8 channels, used on 3D tensors
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1) #8 to 16 channels, same size
        self.batchnorm2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm3d(32)
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm3d(64)
        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm3d(128)
        self.pool = nn.AdaptiveAvgPool3d(1) #reduces the dims the input tensor to 1 value in each dim (takes avg in each spatial dim)->output shape: (batch_size, num_channels, 1, 1, 1)
        self.fc1 = nn.Linear(128, 512) #linear transform for going from 128 input values to 512 output values, weight matrix of size (512, 128) and bias vector of size (512,1)
        self.fc2 = nn.Linear(512, num_parameters) #from 512 input values to num_parameters output values

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x))) #from 1 channel to 8, keep same size, then normalize, then relu
        x = F.max_pool3d(x, kernel_size=2, stride=2) #divide size by 2
        x = F.relu(self.batchnorm2(self.conv2(x))) #from 8 to 16 channels
        x = F.max_pool3d(x, kernel_size=2, stride=2) #divide size by 2
        x = F.relu(self.batchnorm3(self.conv3(x))) #16 to 32 channels
        x = F.max_pool3d(x, kernel_size=2, stride=2) #divide size by 2
        x = F.relu(self.batchnorm4(self.conv4(x))) #32 to 64 channels
        x = F.max_pool3d(x, kernel_size=2, stride=2) #divide size by 2
        x = F.relu(self.batchnorm5(self.conv5(x))) #64 to 128 channels
        x = self.pool(x) #take average on the 3dims so get only 128 values (output shape: (batch_size, num_channels=128, 1, 1, 1))
        x = x.view(x.size(0), -1) #convert x to 2D mat with: n_rows=x.size(0)=batch_size; n_columns=num of all values in the other dims (-> so all the vals for each sample are aligned in a single row)
        x = F.relu(self.fc1(x)) #lin transf for going from 128 to 512 values (multiplied to each row of x, which are all the resulting values of each sample in the batch)
        x = self.fc2(x) #lin transf for going from 512 to num_parameters values
        x = IntervalNormalizationLayer()(x) #scale each value in [ai, bi]
        
        func = ODEFunc(x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[0][5], x[0][6], x[0][7], x[0][8]).to(device)

        pred_y = odeint(func, true_y0, t, atol=1e-6, rtol=1e-6).to(device)

        sim_edv = pred_y[int(func.Tc * args.data_size / args.int_time)][0][0] + 10 #pred_y[n][0][m]: n frame, m-th component (m=0: Vlv-Vd)
        sim_esv = pred_y[int((0.2+0.15*func.Tc)*args.data_size / args.int_time)][0][0] + 10
        sim_ef = (pred_y[int(func.Tc * args.data_size / args.int_time)][0][0] - pred_y[int((0.2+0.15*func.Tc)*args.data_size / args.int_time)][0][0]) / pred_y[int(func.Tc * args.data_size / args.int_time)][0][0] * 100.
        
        return sim_edv, sim_esv, sim_ef, x


#next 2 classes: new (for the ode solver)
class ODEFunc(nn.Module): #takes y, t and returns dydt = NN(y,t). NN: defined in self.net (NN: (y**3, t**3):2D->50->tanh->2D=dydt)
  
    def __init__(self, Tc, Rs, Rm, Ra, Rc, Ca, Cs, Cr, Ls): #cnn: NEW3DCNN
        super(ODEFunc, self).__init__()

        self.Tc = Tc
        self.Rs = Rs
        self.Rm = Rm
        self.Ra = Ra
        self.Rc = Rc
        self.Ca = Ca
        self.Cs = Cs
        self.Cr = Cr
        self.Ls = Ls
        self.Emax = 2.00
        self.Emin = 0.06
    
    def forward(self, t, y):
        dydt = torch.zeros_like(y)

        #self.Rs = self.cnn.Rs

        self.tn =  t / (0.2+0.15*self.Tc) #(t-int(t/self.Tc)*self.Tc) for more loops
        self.P_lv = ((self.Emax-self.Emin)*1.55*(self.tn/0.7)**1.9/((self.tn/0.7)**1.9+1)*1/((self.tn/1.17)**21.9+1) + self.Emin) * y[0][0]

        dydt[0][0] = max(y[0][1]-self.P_lv, 0) / self.Rm - max(self.P_lv-y[0][3], 0) / self.Ra
        dydt[0][1] = (y[0][2]-y[0][1]) / (self.Rs*self.Cr) - max(y[0][1]-self.P_lv, 0) / (self.Cr*self.Rm)
        dydt[0][2] = (y[0][1]-y[0][2]) / (self.Rs*self.Cs) + y[0][4] / self.Cs
        dydt[0][3] = -y[0][4]/self.Ca + max(self.P_lv-y[0][3], 0) / (self.Ca*self.Ra)
        dydt[0][4] = (y[0][3] - y[0][2] - self.Rc * y[0][4])/self.Ls

        return dydt

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

"""
An example usage of the model
"""
patient0 = train_data[0]
seq = patient0["2CH_sequence"]

model = NEW3DCNN(num_parameters = 9) #changed from 8 to 12
# specify the number of initial conditions and parameters into the PVloop (n_parameters = 12)
'''
input = padding(seq, 30, 0)
# perform padding before feeding into NN (dimension 30 since the maximum number of frames is 30)
input_tensor = torch.from_numpy(input).unsqueeze(0).unsqueeze(1).float() # add batch dim at pos 0 (with unsqueeze(0)) and channel dim at pos 0 (with unsqueeze(1))
model(input_tensor) #returns the estimated num_parameters values of the LPmodel
'''
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

ii = 0

model = NEW3DCNN(num_parameters = 9)

#0.97: controls how much weight given to the most recent iterations (larger -> more weight to recent values)
time_meter = RunningAverageMeter(0.97) #measures avg time for running each batch
loss_meter = RunningAverageMeter(0.97) #keep track of avg loss or error during the same iterations

end = time.time()

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the data loader
train_loader = DataLoader(train_data, batch_size=1, shuffle=True) #1 sample (sequence) in each batch

# Training
num_epochs = 5
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        seq = batch["2CH_sequence"]
        
        input = padding(seq, 30, 1) #make the sequence into a 30 frame sequence (add imgs with all 0's, etc)
        input_tensor = torch.from_numpy(input).unsqueeze(0).unsqueeze(1).float() 
        
        #real values
        trueV_ED = torch.tensor([float(batch['info_2CH']['LVedv'][0])])
        trueV_ES = torch.tensor([float(batch['info_2CH']['LVesv'][0])])
        trueEF = torch.tensor([float(batch['info_2CH']['LVef'][0])])

        #simulated values: (and x all the parameters of the circuit estimated by the cnn)
        sim_edv, sim_esv, sim_ef, x = model(input_tensor)

        loss = criterion(sim_edv, trueV_ED) + criterion(sim_esv, trueV_ES) + criterion(sim_ef, trueEF)
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if epoch % 10 == 0:
          
          print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
          
          func = ODEFunc(x[0][0].detach(), x[0][1].detach(), x[0][2].detach(), x[0][3].detach(), x[0][4].detach(), x[0][5].detach(), x[0][6].detach(), x[0][7].detach(), x[0][8].detach()).to(device) #func has a NN: 2->50->tanh->2, and forward(t, y): return NN(y**3)

          pred_y = odeint(func, true_y0, t, atol=1e-6, rtol=1e-6) #here solve the function from the true_y0 and during the whole time t (and not from random points and short durations batch_time as before)
          vi = pred_y[:, :, 0].squeeze().tolist()
          ti = t.squeeze().tolist()
          plt.plot(ti, vi, 'b--')

          # Add axis labels and title
          plt.xlabel('t')
          plt.ylabel('Vlv-Vd')
          plt.axvline(x=func.Tc.detach().numpy(), color='r', linestyle='--') #t_ed
          plt.axvline(x=0.2+0.15*func.Tc.detach().numpy(), color='g', linestyle='--') #t_es

          # Show the plot
          plt.show()

        end = time.time()
        
