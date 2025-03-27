import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader_image import DataLoader_spectral
# 设置数据路径和要加载的能量图像类别
img_dir = "C:/Users/2485636/OneDrive - University of Dundee/Desktop/yuhan-test/yuhantraining_data"
energy_list = ["highkVpImages", "lowkVpImages"]

# 设定 transform（可选）
my_transforms = transforms.Compose([
    transforms.ToTensor()  # 转换为 PyTorch Tensor
])

# 创建 DataLoader
dataset = DataLoader_spectral(img_dir=img_dir, energy_list=energy_list, transform=my_transforms)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 显示 10 组数据
num_samples = 10
for i, (imgs, index) in enumerate(data_loader):
    if i >= num_samples:
        break  # 只显示 10 组数据

    imgs = imgs.squeeze(0).numpy()  # 移除 batch 维度 (shape: [4, 512, 512])

    # 创建子图
    fig, axes = plt.subplots(1, len(energy_list), figsize=(15, 5))
    fig.suptitle(f"Sample {i+1} (Index: {index.item()})", fontsize=14)  # 标题

    # 遍历每个通道，显示不同能量的图像
    for j, energy in enumerate(energy_list):
        axes[j].imshow(imgs[j], cmap="gray")  # 用灰度图显示
        axes[j].set_title(energy)
        axes[j].axis("off")

    plt.show()