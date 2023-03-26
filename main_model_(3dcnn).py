# -*- coding: utf-8 -*-
"""main model (3DCNN).ipynb

# main model (3DCNN)
"""

from google.colab import drive
drive.mount('/content/drive')

#!pip install SimpleITK
import SimpleITK as sitk
import matplotlib.pylab as plt
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import os
from skimage.transform import rescale, resize
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
from scipy.interpolate import interp1d

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

def padding(sequence, n, size=(256,256)):
    n_frames, height, width = sequence.shape
    ratio = height / width

    # Resize each frame to have the same size.
    new_frames = []
    for frame in sequence:
        new_frame = resize(frame, size, mode='constant', anti_aliasing=True)
        new_frames.append(new_frame)
    new_sequence = np.stack(new_frames, axis=0)

    # Pad or truncate the sequence to have 10 frames.
    if n_frames < n:
        padding = np.zeros((n - n_frames, size[0], size[1]))
        new_sequence = np.concatenate([new_sequence, padding], axis=0)
    elif n_frames > n:
        new_sequence = new_sequence[:n]

    return new_sequence

import torch.nn.functional as F

class NEW3DCNN(nn.Module):
    def __init__(self, num_parameters):
        super(NEW3DCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm3d(16)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm3d(32)
        self.conv4 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm3d(64)
        self.conv5 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.batchnorm5 = nn.BatchNorm3d(128)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, num_parameters)
        
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
        return x

import torch
import torch.nn as nn

class CNN3D(nn.Module):
    def __init__(self, n_parameters):
        super(CNN3D, self).__init__()
        self.n_parameters = n_parameters

        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc1 = nn.Linear(128 * 8 * 8 * 2, 512)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(512, self.n_parameters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(-1, 128 * 8 * 8 * 2)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout1(x)

        x = self.fc2(x)

        return x

"""
An example usage of the model
"""
patient0 = train_data[0]
seq = patient0["2CH_sequence"]

model = NEW3DCNN(num_parameters = 8)
# specify the number of initial conditions and parameters into the PVloop (n_parameters = 8)
input = padding(seq, 30)
# perform padding before feeding into NN (dimension 30 since the maximum number of frames is 30)
input_tensor = torch.from_numpy(input).unsqueeze(0).unsqueeze(1).float() # add batch and channel dimensions
model(input_tensor)

import torch
from torch.utils.data import DataLoader
from torch import nn, optim

model = NEW3DCNN(num_parameters = 8)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create the data loader

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

# Training
num_epochs = 5
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):

        seq = batch["2CH_sequence"]
        input = padding(seq, 30)
        input_tensor = torch.from_numpy(input).unsqueeze(0).unsqueeze(1).float() 
        output = model(input_tensor) 

        ## loss1: EF from pv loop and real EF
        trueEF = batch['info_2CH']['LVef']
        # EF_sim = EF_from_PVsimulator(output)
        # loss1 = criterion(trueEF, EF_sim)

        ## loss2: volumes over time

        ## loss3: ED and ES volumes
        true_ED = batch['info_2CH']['LVedv']
        true_ES = batch['info_2CH']['LVesv']

        # loss = loss1+loss2+loss3


        # Backward pass and optimize
        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()

        # Print progress
        if i % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
