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
from autoenco import Autoencoder
from autoenco import CustomDataset


## clean

train_data = np.load('../../../Adversarial-Bone-Length-Attack-on-Action-Recognition-main/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/st-gcn-processed-data/data/NTU-RGB-D/xview/train_data.npy',allow_pickle=True)
val_data = np.load('../../../Adversarial-Bone-Length-Attack-on-Action-Recognition-main/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/st-gcn-processed-data/data/NTU-RGB-D/xview/val_data.npy',allow_pickle=True)



# np.save('../first_frame.npy', my_data_clean[33][:,:,:,:])
## clean




with open('../../../Adversarial-Bone-Length-Attack-on-Action-Recognition-main/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/st-gcn-processed-data/data/NTU-RGB-D/xview/train_label.pkl', 'rb') as f:
  sample_name, train_label = pickle.load(f)
with open('../../../Adversarial-Bone-Length-Attack-on-Action-Recognition-main/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/st-gcn-processed-data/data/NTU-RGB-D/xview/train_label.pkl', 'rb') as f:
  sample_name, val_label = pickle.load(f)


train_label = np.array(train_label)
val_label = np.array(val_label)

train_dataset = CustomDataset(train_data, train_label)
val_dataset = CustomDataset(val_data, val_label)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

model = Autoencoder()
model.to('cuda:0')
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for data, label in tqdm(train_dataloader):
        with torch.no_grad():
          data = data.float().cuda(0)
          label = label.long().cuda(0)
          
          
        optimizer.zero_grad()
        outputs = model(data)
        
        
        
        
        
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    
    print('validation starts')
    
    model.eval()
    val_loss = 0.0
    best_val_loss = 1000000
    with torch.no_grad():
        for data, label in tqdm(val_dataloader):
            data = data.float().cuda(0)
            label = label.long().cuda(0)
            outputs = model(data)
            loss = criterion(outputs, data)
            val_loss += loss.item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), './weights/best_autoencoder.pth')
        print(f"Saved Best Model with Val Loss: {best_val_loss}")    
           
print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_dataloader)}, Val Loss: {val_loss / len(val_dataloader)}')
        
        