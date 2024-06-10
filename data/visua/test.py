import torch
from torch.utils.data import DataLoader
from autoenco import Autoencoder
from autoenco import CustomDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm
# Assuming you already have a CustomDataset class defined and test data loaded
val_data = np.load('../../../Adversarial-Bone-Length-Attack-on-Action-Recognition-main/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/st-gcn-processed-data/data/NTU-RGB-D/xview/val_data.npy',allow_pickle=True)

with open('../../../Adversarial-Bone-Length-Attack-on-Action-Recognition-main/Adversarial-Bone-Length-Attack-on-Action-Recognition-main/st-gcn-processed-data/data/NTU-RGB-D/xview/train_label.pkl', 'rb') as f:
  sample_name, val_label = pickle.load(f)
val_label = np.array(val_label)
test_dataset = CustomDataset(val_data, val_label)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize your model
model = Autoencoder()
model.to('cuda:0')

# Load the saved weights
model.load_state_dict(torch.load('./weights/best_autoencoder.pth'))

# Function to evaluate the model on test data
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for data, _ in tqdm(dataloader):
            data = data.float().cuda(0)
            outputs = model(data)
            data = data.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            np.save('results_ae/input.npy', data)
            np.save('results_ae/ouput.npy', outputs)
            print(data.shape)
            print(outputs.shape)
            exit()
            loss = criterion(outputs, data)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Calculate the loss on the test set
test_loss = evaluate_model(model, test_dataloader)
print(f'Test Loss: {test_loss}')