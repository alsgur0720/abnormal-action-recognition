import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#from .visualizations import Ax3DPose
import sys
from einops import rearrange
import pickle
import torch
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 인코더 정의
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # [batch, 16, 150, 13]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [batch, 32, 75, 7]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [batch, 64, 38, 4]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * 38 * 4, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 128),
            nn.LeakyReLU()
        )
        
        # 디코더 정의
        self.decoder = nn.Sequential(
            nn.Linear(128, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 64 * 38 * 4),
            nn.LeakyReLU(),
            nn.Unflatten(1, (64, 38, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 32, 75, 8]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [batch, 16, 150, 16]
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),  # [batch, 3, 300, 25]
            nn.Tanh()
        )
        
        
        self.data_bn = nn.BatchNorm1d(2 * 3 * 25) #num_person * in_channels * num_point

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        data_motion = x[:,:,1:,:] - x[:,:,:-1,:]
        encoded = self.encoder(data_motion)
        decoded = self.decoder(encoded)
        
        restored_skeleton = decoded[:N * M, :C, :T, :V].clone()
        
        restored_skeleton[:,:,0,:] = x[:,:,0,:]
        
        for j in range(0, N * M):
          for i in range(0, 299):
            if (torch.all(data_motion[j,:,i,:] == 0)) == True:
              restored_skeleton[j,:,i,:] = 0
            else:
              restored_skeleton[j,:,i+1,:] = restored_skeleton[j,:,i,:] + restored_skeleton[j,:,i+1,:]
        
        restored_skeleton = restored_skeleton.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        
        return restored_skeleton


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        """
        data: 데이터 배열 (예: [18932, 3, 200, 25, 2] 형태의 텐서)
        labels: 라벨 배열 (예: [18932] 형태의 텐서)
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        """
        데이터셋의 전체 데이터 수 반환
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        주어진 인덱스 idx에 해당하는 데이터와 라벨을 반환
        """
        return self.data[idx], self.labels[idx]