# -*- coding: utf-8 -*-

# main model (3DCNN)
## ref: https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python

!pip install SimpleITK
import SimpleITK as sitk
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

batch_size = 50

## Functions for Dataloading"""

class CamusIterator(Dataset):
    def __init__( 
        self, 
        data_type='train', 
        global_transforms=[], 
        augment_transforms=[]
    ):
        super(CamusIterator, self).__init__()
        
        path = '/in-silico-ML2/CAMUS_public/database_nifti'
        train_file='/in-silico-ML/CAMUS_public/database/training'
        test_file='/in-silico-ML/CAMUS_public/database/testing'
        
        if data_type == 'train':
            data_file = path
        elif data_type == 'test':
            data_file = path
        else:
            raise Exception('Wrong data_type for CamusIterator')
            
        self.data_type = data_type
        self.data_file = data_file
        self.global_transforms = global_transforms
        self.augment_transforms = augment_transforms
    
    def __read_image( self, patient_file, suffix ):
        image_file = '{}/{}/{}'.format( self.data_file, patient_file, patient_file+suffix )
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

        info_2CH = self.__read_info( '{}/{}/{}'.format(self.data_file, patient_file, 'Info_2CH.cfg') )
        #info_4CH = self.__read_info( '{}/{}/{}'.format(self.data_file, patient_file, 'Info_4CH.cfg') )
        
        if self.data_type == 'train':
            data = {
                'patient': patient_file,
                'info_2CH': info_2CH,    # Dictionary of infos
                #'info_4CH': info_4CH    # Dictionary of infos
            }
        elif self.data_type == 'test':
            data = {
                'patient': patient_file,
                'info_2CH': info_2CH,   # Dictionary of infos
                #'info_4CH': info_4CH   # Dictionary of infos
            }
        
        # Transforms
        for transform in self.global_transforms:
            data = transform(data)
        for transform in self.augment_transforms:
            data = transform(data)
            
        return data

    def __iter__( self ):
        for i in range( len(self) ):
            yield self[ i ]

class ResizeImagesAndLabels(object):
    
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

class ResizeSeq(object):
    
    def __init__(self, n, fields=['2CH_sequence']):
        self.n = n
        self.fields = fields
        
    def __call__(self, data):
        for field in self.fields:            
            # transpose to go from chan x h x w to h x w x chan and back.
            data[field] = padding(data[field], n = self.n)     

        return data


global_transforms = [
]
augment_transforms = [
]

train_data = CamusIterator(
    data_type='train',
    global_transforms=global_transforms,
    augment_transforms=augment_transforms,
)
train_data = Subset(train_data, range(500))

info_data = []
data_batch_size = 10
ES = []
FR = []
path = '/in-silico-ML/input'


# Loop over each patient in train_data
for i in range(len(train_data)):  
    ES.append(train_data[i]['info_2CH']['ES'])
    FR.append(train_data[i]['info_2CH']['FrameRate'])
    # Extract info data and add to list
    #info_data.append(train_data[i]['info_2CH'])

ES = np.array(ES)
FR = np.array(FR)

ES = ES.astype(float)
FR = FR.astype(float)

delta = (ES-1)*1/FR

df = pd.DataFrame(delta, columns=['times'])

df.to_csv(path + '/vs_time.csv', index=False)
