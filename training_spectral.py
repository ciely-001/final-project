import sys

sys.path.append("/DPS")
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from dataloader_spectral import DataLoader_spectral
from torch.optim import Adam
from myutils_ddpm import loss_fn, diffusion_parameters
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# CHECKLIST :
# - chkpt name (checkpoint name to save the weights of the NN)
# - lr (learning rate and scheluder)
# - epochs
# - logdir (for tensorboard if needed)
# - batch_size
# - pixel_size (in order to convert into pixel^(-1))


# Transforms to apply to images (resize, data augmentation, ..)
my_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5)]
)

# 训练数据 & 验证数据
energy_list = ['highkVpTransmission', 'lowkVpTransmission']

data_train = DataLoader_spectral(img_dir='C:/Users/2485636/OneDrive - University of Dundee/Desktop/yuhan-test/yhtraining_data/', energy_list=energy_list, transform=my_transforms)
data_val = DataLoader_spectral(img_dir='C:/Users/2485636/OneDrive - University of Dundee/Desktop/yuhan-test/yhvalidation_data/', energy_list=energy_list, transform=my_transforms)
pixel_size = 1
batch_size = 4 # 32
Data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
Data_loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=True)

# 计算数据的均值和标准差
mean = torch.tensor(np.load('C:/Users/2485636/OneDrive - University of Dundee/Desktop/yuhan-test/mean-std-data/mean_spectral.npy'))[None,:,None,None]
std = torch.tensor(np.load('C:/Users/2485636/OneDrive - University of Dundee/Desktop/yuhan-test/mean-std-data/std_spectral.npy'))[None,:,None,None]


# U-NET
from Unet01 import UNet
n_bins = len(energy_list)
ddpm_model = UNet(image_channels=n_bins, n_channels=8)
ddpm_model = ddpm_model.to(device)
ckpt_name = "./checkpoints_spectral/nn_weights/spectral.pth"
# If poursuing a previous training :
# ddpm_model.load_state_dict(torch.load(ckpt_name))


# # Optimisation parameters
lr = 1e-3
n_epochs = 400

optimizer = Adam(ddpm_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
writer = SummaryWriter(log_dir="./checkpoints_spectral/runs/")

# # DIFFUSION PARAMETERS & FUNCTIONS

T = 1000
alpha, alpha_bar = diffusion_parameters(T)
alpha_bar = alpha_bar.to(device).requires_grad_(False)

for epoch in range(n_epochs):

    avg_loss = 0.
    num_items = 0

    avg_loss_val = 0.
    num_items_val = 0

    for x, index in Data_loader_train:
        x = x * pixel_size
        x = (x - mean) / std
        x = x.float()

        x = x.to(device).requires_grad_(False)
        loss = loss_fn(ddpm_model, x, T, alpha_bar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    with torch.no_grad():
        for x_val, index in Data_loader_val:
            x_val = x_val * pixel_size
            x_val = (x_val - mean) / std
            x_val = x_val.float()
            x_val = x_val.to(device).detach().requires_grad_(False)

            loss = loss_fn(ddpm_model, x_val, T, alpha_bar)

            avg_loss_val += loss.item() * x_val.shape[0]
            num_items_val += x_val.shape[0]

    writer.add_scalars('run', {'Train': avg_loss / num_items,
                               'Val': avg_loss_val / num_items_val}, epoch)
    scheduler.step()

    if epoch % 10 == 0:
        print(epoch, avg_loss / num_items)
        torch.save(ddpm_model.state_dict(), ckpt_name)

# Update the checkpoint at the end of training.
torch.save(ddpm_model.state_dict(), ckpt_name)
writer.flush()