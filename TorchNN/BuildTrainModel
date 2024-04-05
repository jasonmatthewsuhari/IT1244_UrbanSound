import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
import torch
from torch.utils.data import random_split
from DataLoad import *
import torch
import pandas as pd
import os
from AudioNN import *

data_dir = 'Train'
classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
data_path = '/Users/Joshua/Documents/IT1244 AI/Dataset/Audio Dataset/Urban Sound/' #replace local data path to training here
model_path = '/Users/Joshua/Documents/IT1244 AI/Dataset/Audio Dataset/Urban Sound/Torch/model.pth' #replace stored model path here

df = createAudioFrame('Train',classes)
myds = SoundDS(df,data_path)

num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)
# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device
num_epochs=35
training(myModel, train_dl, num_epochs,device)
# Run inference on trained model with the validation set
inference(myModel, val_dl,device)
torch.save(myModel.state_dict(), model_path)
