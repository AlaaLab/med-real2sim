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
import collections
import pandas
import skimage.draw
import torchvision
import echonet

#odesolver:
from torch.storage import T
import argparse
import time

batch_size = 40
# data_size = 450
num_epochs = 200
# val_size = 50
learning_rate = 0.005
ID = '306_full_echonet_morelabels'

file = f"{ID}_epoch_{num_epochs}_lr_{learning_rate}"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sequences_all = []
info_data_all = []
path = '/scratch/users/keying_kuang/ML/echonet/EchoNet-Dynamic'
output_path = '/accounts/biost/grad/keying_kuang/ML/interpolator_RaRm_10'

class Echo(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.
    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_type (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """
    def normalize_and_interpolate_frames(self, video, target_frames=10):
        """
        Normalize and interpolate frames of a video to a specified number of target frames.

        Args:
        - video (np.ndarray): Input video array of shape (num_frames, channels, height, width).
        - target_frames (int): Number of target frames for interpolation.

        Returns:
        - interpolated_video (torch.Tensor): Interpolated video tensor of shape (target_frames, channels, height, width).
        """
        # Convert NumPy array to PyTorch tensor
        video = torch.from_numpy(video)

        num_frames, channels, height, width = video.shape

        # Calculate indices for interpolation
        indices = torch.linspace(0, num_frames - 1, target_frames)

        # Get the integer and fractional parts of the indices
        indices_floor = indices.floor().long()
        indices_frac = indices - indices_floor.float()

        # Perform linear interpolation
        interpolated_video = (
            video[indices_floor] * (1 - indices_frac.view(-1, 1, 1, 1)) +
            video[torch.clamp(indices_floor + 1, 0, num_frames - 1)] * indices_frac.view(-1, 1, 1, 1)
        )

        return interpolated_video
    def __init__(self, root=None,
                 split="train", target_type="EF",
                 mean=0., std=1.,
                 length=16, period=2,
                 max_length=250,
                 clips=1,
                 pad=None,
                 noise=None,
                 target_transform=None,
                 external_test_location=None):
        #if root is None:
            # root = echonet.config.DATA_DIR

        super().__init__(root, target_transform=target_transform)

        self.split = split.upper()
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = target_type
        self.mean = mean
        self.std = std
        self.length = length
        self.max_length = max_length
        self.period = period
        self.clips = clips
        self.pad = pad
        self.noise = noise
        self.target_transform = target_transform
        self.external_test_location = external_test_location

        self.fnames, self.outcome = [], []

        if self.split == "EXTERNAL_TEST":
            self.fnames = sorted(os.listdir(self.external_test_location))
        else:
            # Load video-level labels
            with open(os.path.join(self.root, "FileList.csv")) as f:
                data = pandas.read_csv(f)
            data["Split"].map(lambda x: x.upper())

            if self.split != "ALL":
                data = data[data["Split"] == self.split]

            self.header = data.columns.tolist()
            self.fnames = data["FileName"].tolist()
            self.fnames = [fn + ".avi" for fn in self.fnames if os.path.splitext(fn)[1] == ""]  # Assume avi if no suffix
            self.outcome = data.values.tolist()

            # Check that files are present
            missing = set(self.fnames) - set(os.listdir(os.path.join(self.root, "Videos")))
            if len(missing) != 0:
                print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
                for f in sorted(missing):
                    print("\t", f)
                raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

            # Load traces
            self.frames = collections.defaultdict(list)
            self.trace = collections.defaultdict(_defaultdict_of_lists)

            with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
                header = f.readline().strip().split(",")
                assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

                for line in f:
                    filename, x1, y1, x2, y2, frame = line.strip().split(',')
                    x1 = float(x1)
                    y1 = float(y1)
                    x2 = float(x2)
                    y2 = float(y2)
                    frame = int(frame)
                    if frame not in self.trace[filename]:
                        self.frames[filename].append(frame)
                    self.trace[filename][frame].append((x1, y1, x2, y2))
            for filename in self.frames:
                for frame in self.frames[filename]:
                    self.trace[filename][frame] = np.array(self.trace[filename][frame])

            # A small number of videos are missing traces; remove these videos
            keep = [len(self.frames[f]) >= 2 for f in self.fnames]
            self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
            self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]

    def __getitem__(self, index):
        # Find filename of video
        if self.split == "EXTERNAL_TEST":
            video = os.path.join(self.external_test_location, self.fnames[index])
        elif self.split == "CLINICAL_TEST":
            video = os.path.join(self.root, "ProcessedStrainStudyA4c", self.fnames[index])
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])

        # Load video into np.array
        video = echonet.utils.loadvideo(video).astype(np.float32)
        
        # Add simulated noise (black out random pixels)
        # 0 represents black at this point (video has not been normalized yet)

        # Set number of frames
        c, f, h, w = video.shape
        if self.length is None:
            # Take as many frames as possible
            length = f // self.period
        else:
            # Take specified number of frames
            length = self.length

        if self.max_length is not None:
            # Shorten videos to max_length
            length = min(length, self.max_length)

        if f < length * self.period:
            # Pad video with frames filled with zeros if too short
            # 0 represents the mean color (dark grey), since this is after normalization
            video = np.concatenate((video, np.zeros((c, length * self.period - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633
        key = self.fnames[index]
        smallindex = self.frames[key][0]
        largeindex = self.frames[key][-1]
        index_difference = largeindex - smallindex

        if index_difference < 10:
        # Exclude videos with index difference less than 10
            return None, None    

        if self.clips == "all":
            # Take all possible clips of desired length
            start = np.arange(f - (length - 1) * self.period)
        else:
            # Take random clips from video
            start = np.random.choice(f - (length - 1) * self.period, self.clips)

        # Gather targets
        target = []
        for t in self.target_type:
            key = self.fnames[index]
            if t == "Filename":
                target.append(self.fnames[index])
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target.append(int(self.frames[key][-1]))
            elif t == "SmallIndex":
                # Largest (diastolic) frame is first
                target.append(int(self.frames[key][0]))
            elif t == "LargeFrame":
                target.append(video[:, self.frames[key][-1], :, :])
            elif t == "SmallFrame":
                target.append(video[:, self.frames[key][0], :, :])
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    t = self.trace[key][self.frames[key][-1]]
                else:
                    t = self.trace[key][self.frames[key][0]]
                x1, y1, x2, y2 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(int), np.rint(x).astype(int), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target.append(mask)
            else:
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target.append(np.float32(0))
                else:
                    target.append(np.float32(self.outcome[index][self.header.index(t)]))

        if target != []:
            target = tuple(target) if len(target) > 1 else target[0]
            if self.target_transform is not None:
                target = self.target_transform(target)

        # Select clips from video
        smallindex = self.frames[key][0]
        largeindex = self.frames[key][-1]
        selected_frames = np.arange(smallindex, largeindex + 1, 1)  # Include all frames in the specified range

        # Gather the frames and form the video
        video = tuple(video[:, s, :, :] for s in selected_frames)
        if self.clips == 1:
            video = video[0]
        else:
            video = np.stack(video)
            normalized_interpolated_video = self.normalize_and_interpolate_frames(video, target_frames=10)

        if self.pad is not None:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
            i, j = np.random.randint(0, 2 * self.pad, 2)
            video = temp[:, :, i:(i + h), j:(j + w)]

        return normalized_interpolated_video, target

    def __len__(self):
        return len(self.fnames)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.
    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)
## 
print("Done loading training data!")
# define normalization layer to make sure output xi in an interval [ai, bi]:
# define normalization layer to make sure output xi in an interval [ai, bi]:
def collate_fn(batch):
    # Filter out samples with None values for both video and target
    batch = [(video, target) for video, target in batch if video is not None or target is not None]

    # Separate videos and targets
    videos, targets = zip(*batch)

    # Stack videos if there are any
    videos = torch.stack(videos) if videos and any(v is not None for v in videos) else None

    # Stack targets if there are any, handling the case where an element is an integer
    targets = tuple(torch.stack(t, dim=0) if t is not None and isinstance(t[0], torch.Tensor) else torch.tensor(t) for t in zip(*targets)) if targets else None

    return videos, targets

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
        
        self.conv1 = nn.Conv3d(3, 8, kernel_size=3, padding=1)
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
        self.norm1 = IntervalNormalizationLayer()
        
    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        # print("Input size:", x.size())
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        # print("Input size:", x.size())
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        # print("Input size:", x.size())
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # print("Input size:", x.size())
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
        self.fc1 = nn.Linear(6, 256).double()
        self.fc2 = nn.Linear(256, 10).double()
    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = self.fc2(z)
        return z

# Initialize the neural network
net = Interpolator()



net.load_state_dict(torch.load('/accounts/biost/grad/keying_kuang/ML/interpolator_RaRm_10/interpRaRm_param_weight_2_EStoED.pt'))
print("Done loading interpolator!")

model = NEW3DCNN(num_parameters = 7)
model.to(device)
# model.load_state_dict(torch.load('/accounts/biost/grad/keying_kuang/ML/interpolator2/001_weight_model_NNinterp2_allloss.pt'))

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create the data loader
train_data = Echo(
    root=path,
    split="train",  # Set the split (train, val, test, etc.)
    target_type=["SmallIndex", "LargeIndex", "ESV", "EDV"],  # Specify the target type(s) you need
    length=None,  # Set the length to None for including all frames
    period=1,  # Set the period to 1 for every frame
    clips= None,
    pad=None,  # Set the padding if needed
    noise=None,  # Set the noise level if needed
    target_transform=None,  # Set the target transform if needed
    external_test_location=None # Set the external test location if needed
)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
print("Done creating dataloaders ")

# loading validation set
validation_data = Echo(
    root=path,
    split="val",  # Set the split (train, val, test, etc.)
    target_type=["EF", "ESV", "EDV"],  # Specify the target type(s) you need
    length=None,  # Set the length to None for including all frames
    period=1,  # Set the period to 1 for every frame
    clips= None,
    pad=None,  # Set the padding if needed
    noise=None,  # Set the noise level if needed
    target_transform=None,  # Set the target transform if needed
    external_test_location=None # Set the external test location if needed
)
val_loader = DataLoader(validation_data, batch_size=40, shuffle=True, collate_fn=collate_fn)
print(len(validation_data))


val_data = next(iter(val_loader))
val_seq = val_data[0]
val_tensor = torch.tensor(val_seq, dtype=torch.float32) 
val_tensor = val_tensor.permute(0, 2, 1, 3, 4)
val_EF = val_data[1][0]

print("Done loading validation set!")

class UNet(nn.Module):
    def __init__(self, in_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)  # Output a single channel
        )

        # Fully connected layers for regression
        self.fc1 = nn.Linear(56 * 56, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)

        # Middle
        x2 = self.middle(x1)

        # Decoder
        x3 = self.decoder(x2)

        # Flatten for fully connected layers
        x4 = x3.view(x3.size(0), -1)

        # Fully connected layers
        x5 = self.fc1(x4)
        x6 = self.fc2(x5)

        return x6

# Assuming input frames have 3 channels (RGB)
modelu = UNet(in_channels=3)
modelu.load_state_dict(torch.load('/accounts/biost/grad/keying_kuang/ML/interpolator_RaRm_10/best_unet_model_0.001.pt'))
print("Done loading interpolator!")

# Training
num_epochs = num_epochs
best_mae = float('inf')
patience = 15

#lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=patience, verbose=True)
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for j, batch in enumerate(train_loader):
        optimizer.zero_grad()
        seq = batch[0]
        input_tensor = seq.to(torch.float32)
        input_tensor = input_tensor.permute(0, 2, 1, 3, 4)

        #simulated values: sim_output = (V_ED, V_ES)
        x = model(input_tensor).double()
        x1 = x[:, :6]
        Vd = x[:, -1:]
        output1 = net(x1)
        output = output1 + Vd-4

        output_labels = torch.zeros((input_tensor.shape[0], 10), dtype = torch.double)
        input_sequence = seq
        # Iterate over the batch dimension
        for batch_idx in range(input_sequence.size(0)):
            # Iterate over the frame dimension
            for frame_idx in range(input_sequence.size(1)):
                # Extract a single frame
                frame = input_sequence[batch_idx, frame_idx]

                # Assuming your model expects a single frame as input
                frame = frame.unsqueeze(0)  # Add batch dimension
                # Pass the frame through the model to get the label
                label = modelu(frame)

        # Store the label in the output tensor
                output_labels[batch_idx, frame_idx] = label.item()


        #criterion = torch.nn.MSELoss()
        loss = criterion(output_labels, output)
        #loss = criterion(ved, trueV_ED) + criterion(ves, trueV_ES) + criterion(ef, true_EF)
        #loss = criterion(ef.flatten(), true_EF)
        # Compute the gradients and update the model parameters using backpropagation
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)

    with torch.no_grad():
        val_x = model(val_tensor).double()
        val_x1 = val_x[:, :6]
        val_Vd = val_x[:, -1:]
        val_output1 = net(val_x1)
        val_output = val_output1 + val_Vd -4

        val_input_sequence = val_seq
        val_output_labels = torch.zeros(val_input_sequence.size(0), 10)

        # Iterate over the batch dimension
        for batch_idx in range(val_input_sequence.size(0)):
            # Iterate over the frame dimension
            for frame_idx in range(val_input_sequence.size(1)):
                # Extract a single frame
                frame = val_input_sequence[batch_idx, frame_idx]

                # Assuming your model expects a single frame as input
                frame = frame.unsqueeze(0)  # Add batch dimension
                # Pass the frame through the model to get the label
                val_label = modelu(frame)

        # Store the label in the output tensor
                val_output_labels[batch_idx, frame_idx] = val_label.item()

        b, a2, a3, a4, a5, a6, a7, a8, a9, a = torch.split(val_output, split_size_or_sections=1, dim=1)
        val_sim_EF = (a-b)/a*100
        MAE = criterion(val_output_labels, val_output)
    val_EF_np = val_EF.numpy()
    val_sim_EF_np = val_sim_EF.detach().numpy()
    MAEr = np.mean(np.abs(val_EF_np - val_sim_EF_np.flatten()))
    #lr_scheduler.step(MAE)
    print("Epoch [{}/{}], Loss: {:.4f}, valid MAE: {:.4f}, valid loss: {:.4f}".format(epoch+1, num_epochs, epoch_loss, MAEr, MAE))
    if MAE < best_mae:
        best_mae = MAE
        file_label = f"epoch:{epoch+1} MAT:{MAE:.2f}"
        print(file_label)
        torch.save(model.state_dict(), f'{output_path}/{file}_weight_best_model.pt')
    

torch.save(model.state_dict(), os.path.join(output_path,f'{file}_end_weight.py'))

