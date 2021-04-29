import torch
from torch import nn
# import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision import datasets, transforms
import os
import pathlib
import sys
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import numpy as np
from pytorch_msssim import ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(sys.path[0])))

# if torch.cuda.is_available():
#     print("Using the GPU. You are good to go!")
#     device = torch.device('cuda:0')
# else:
#     raise Exception("WARNING: Could not find GPU! Using CPU only. \
# To enable GPU, please to go Edit > Notebook Settings > Hardware \
# Accelerator and select GPU.")

# -----------------
# MODEL
# -----------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
              
        return x


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans=10, num_pool_layers=4, drop_prob=0.0):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net
                model.
            chans (int): Number of output channels of the first convolution
                layer.
            num_pool_layers (int): Number of down-sampling and up-sampling
                layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, drop_prob)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        ]

    def forward(self, image):
        """
        Args:
            image (torch.Tensor): Input tensor of shape [batch_size,
                self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans,
                height, width]
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image):
        """
        Args:
            image (torch.Tensor): Input tensor of shape [batch_size,
                self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans,
                height, width]
        """
        return self.layers(image)

    def __repr__(self):
        return (
            f"ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, "
            f"drop_prob={self.drop_prob})"
        )


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans, out_chans):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image):
        """
        Args:
            image (torch.Tensor): Input tensor of shape [batch_size,
                self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans,
                height, width]
        """
        return self.layers(image)

    def __repr__(self):
        return f"ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans})"


# ----------------
# TRAIN
# ----------------
print(torch.cuda.current_device())
yo = np.ones((1, 640, 320))
yo2 = yo[0,:,:]
print(yo2.shape)

# ----------------
# DATA
# ----------------
imgs = os.listdir(('train'))
num_imgs = len(imgs)
ground_truth = Image.open('0_equispaced.png')
# print('gt', gt.shape)
gt = np.asarray(ImageOps.grayscale(ground_truth))
# gt = np.resize(gt, (320, 320))
print('gt', gt.shape)
gt_tensor = torch.FloatTensor(gt).unsqueeze(0)
fully_sampled = Image.open('fully_sampled.png')
fs = np.asarray(ImageOps.grayscale(fully_sampled))
# fs = np.resize(fs, (320, 320))
# plt.imsave('fs_resized.png', fs, cmap='gray')
fs_tensor = torch.FloatTensor(fs).unsqueeze(0)
print('initial_ssim', ssim(gt_tensor.unsqueeze(1), fs_tensor.unsqueeze(1), data_range=torch.max(fs_tensor).item()))

X = np.zeros((num_imgs, 640, 320))
Y = np.zeros((num_imgs, 640, 320))
count = 0

for fname in imgs:
    img = Image.open('train/' + fname)
    img = np.asarray(ImageOps.grayscale(img))
    # print(fname)
    # print('shape', img.shape)
    # breakpoint()
    X[count] = img
    Y[count] = gt
    count += 1

X_train = torch.unsqueeze(torch.FloatTensor(X), 1)
Y_train = torch.unsqueeze(torch.FloatTensor(Y), 1)
# X_train = torch.FloatTensor(X)
# Y_train = torch.FloatTensor(Y)

train = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size = 1, shuffle = False)

# ----------------
# OPTIMIZER
# ----------------
model = Unet(1, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------
# LOSS
# ----------------
def SSIM_loss(output, target):
  return 1 - ssim(output.unsqueeze(1), target.unsqueeze(1), data_range=torch.max(target).item())

# ----------------
# TRAINING LOOP
# ----------------
num_epochs = 40
for epoch in range(num_epochs):
  print('epoch', epoch)
  # TRAINING LOOP
  for train_batch in train_loader:
    x, y = train_batch
    output = model(x)
    loss = SSIM_loss(output, y)
    print('train loss: ', loss.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

pred = model(gt_tensor.unsqueeze(0))
print(pred.shape)
# fs_tensor = fs_tensor.unsqueeze(0)
# plt.imsave('pred.png', pred.squeeze(0).detach().numpy())
print('final_ssim', ssim(pred, fs_tensor.unsqueeze(1), data_range=torch.max(fs_tensor).item()))
# plt.imsave('pred.png', pred.squeeze(0).detach().numpy(), cmap='gray')
pred_img = pred.squeeze(0)
print('pred_img1', pred_img.shape)
pred_img = pred_img.squeeze(0)
# print('pred_img2', pred_img.shape)
pred_img = pred_img.detach().numpy()
print('pred_img', pred_img.shape)
plt.imsave('pred.png', pred_img, cmap='gray')