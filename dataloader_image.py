import torch
import numpy as np
from torch.utils.data import Dataset
import os


class DataLoader_spectral(Dataset):

    def __init__(self, img_dir, energy_list, transform=None):
        """
        img_dir: 数据存放的文件夹路径（如 'path/to/training_data/'）
        energy_list: 包含能量相关的 .npy 文件名列表
        transform: 可选的图像变换（如 Tensor 转换等）
        """
        self.img_dir = img_dir
        self.energy_list = energy_list
        self.transform = transform

        # 读取所有 .npy 文件并存入字典
        self.data = {}
        for energy in energy_list:
            file_path = os.path.join(img_dir, f"{energy}.npy")
            if os.path.exists(file_path):
                self.data[energy] = np.load(file_path)
            else:
                raise FileNotFoundError(f"文件 {file_path} 不存在")

        # 确保所有 .npy 文件的样本数相同
        self.n_data = self.data[energy_list[0]].shape[0]

    def __getitem__(self, index):
        """
        加载索引 index 对应的所有能量图像，并拼接成 [512, 512, len(energy_list)]
        """
        imgs = np.zeros([512, 512, len(self.energy_list)], dtype=np.float32)

        for i, energy in enumerate(self.energy_list):
            imgs[:, :, i] = self.data[energy][index]  # 取对应 index 位置的切片

        # 进行可选的 transform 处理（如转换为 Tensor）
        if self.transform:
            imgs = self.transform(imgs)

        return imgs, index  # 返回图像和索引

    def __len__(self):
        return self.n_data
