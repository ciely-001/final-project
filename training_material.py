import sys

sys.path.append("/share/castor/home/vazia/DPS")
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from dataloader_material import DataLoader_material
from torch.optim import Adam
from myutils_ddpm import loss_fn, diffusion_parameters
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# 数据增强
my_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5)
])

# 训练数据 & 验证数据
material_list = ['Phantom_Adipose', 'Phantom_Calcification', 'Phantom_Fibroglandular']

data_train = DataLoader_material(img_dir='C:/Users/2485636/OneDrive - University of Dundee/Desktop/yuhan-test/yhtraining_data/', material_list=material_list, transform=my_transforms)
data_val = DataLoader_material(img_dir='C:/Users/2485636/OneDrive - University of Dundee/Desktop/yuhan-test/yhvalidation_data/', material_list=material_list, transform=my_transforms)

batch_size = 4 # 32
Data_loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
Data_loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=True)

# 计算数据的均值和标准差
mean = torch.tensor(np.load('C:/Users/2485636/OneDrive - University of Dundee/Desktop/yuhan-test/mean-std-data/mean_material.npy'))[None,:,None,None]
std = torch.tensor(np.load('C:/Users/2485636/OneDrive - University of Dundee/Desktop/yuhan-test/mean-std-data/std_material.npy'))[None,:,None,None]

# U-NET
from Unet01 import UNet

n_mat = len(material_list)
ddpm_model = UNet(image_channels=n_mat, n_channels=8)
ddpm_model = ddpm_model.to(device)
ckpt_name = "./checkpoints_material/nn_weights/material.pth"

# 训练参数
lr = 1e-3
n_epochs = 400
optimizer = Adam(ddpm_model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
writer = SummaryWriter(log_dir="./checkpoints_material/runs/")

# DIFFUSION PARAMETERS
T = 1000
alpha, alpha_bar = diffusion_parameters(T)
alpha_bar = alpha_bar.to(device).requires_grad_(False)

# 训练循环
for epoch in range(n_epochs):
    avg_loss = 0.
    num_items = 0

    avg_loss_val = 0.
    num_items_val = 0

    # 训练
    for x, index in Data_loader_train:
        x = (x - mean) / std
        x = x.float().to(device).requires_grad_(False)

        loss = loss_fn(ddpm_model, x, T, alpha_bar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    # 验证
    with torch.no_grad():
        for x_val, index in Data_loader_val:
            x_val = (x_val - mean) / std
            x_val = x_val.float().to(device).detach().requires_grad_(False)

            loss = loss_fn(ddpm_model, x_val, T, alpha_bar)
            avg_loss_val += loss.item() * x_val.shape[0]
            num_items_val += x_val.shape[0]

    writer.add_scalars('run', {
        'Train': avg_loss / num_items,
        'Val': avg_loss_val / num_items_val
    }, epoch)

    scheduler.step()

    if epoch % 10 == 0:
        print(epoch, avg_loss / num_items)
        torch.save(ddpm_model.state_dict(), ckpt_name)

# 训练完成后保存模型
torch.save(ddpm_model.state_dict(), ckpt_name)
writer.flush()
