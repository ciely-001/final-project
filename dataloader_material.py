import torch
import numpy as np
from torch.utils.data import Dataset
import os


class DataLoader_material(Dataset):

    def __init__(self, img_dir, material_list, transform=None):
        """
        img_dir: 数据存放的文件夹路径（如 'path/to/training_data/'）
        material_list: 包含材料名称的列表，例如 ['Phantom_Adipose', 'Phantom_Calcification', 'Phantom_Fibroglandular']
        transform: 可选的图像变换（如 Tensor 转换等）
        """
        self.img_dir = img_dir
        self.material_list = material_list
        self.transform = transform

        # 读取所有材料的.npy文件，并存入字典
        self.data = {}
        for material in material_list:
            file_path = os.path.join(img_dir, f"{material}.npy")
            if os.path.exists(file_path):
                self.data[material] = np.load(file_path)
            else:
                raise FileNotFoundError(f"文件 {file_path} 不存在")

        # 获取数据长度（假设每个材料的.npy文件第一维度是相同的，即样本数相同）
        self.n_data = self.data[material_list[0]].shape[0]

    def __getitem__(self, index):
        """
        加载索引 index 对应的所有材料的图像，并拼接成 [512, 512, len(material_list)]
        """
        imgs = np.zeros([512, 512, len(self.material_list)], dtype=np.float32)

        for i, material in enumerate(self.material_list):
            imgs[:, :, i] = self.data[material][index]  # 取对应 index 位置的切片

        # 进行可选的 transform 处理（如转换为 Tensor）
        if self.transform:
            imgs = self.transform(imgs)

        return imgs, index  # 返回图像和索引

    def __len__(self):
        return self.n_data
