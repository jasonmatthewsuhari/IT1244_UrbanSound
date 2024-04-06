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

data_dir = 'Test'
classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
data_path = '/Users/Joshua/Documents/IT1244 AI/Dataset/Audio Dataset/Urban Sound/' #replace local data path to training here
model_path = '/Users/Joshua/Documents/IT1244 AI/Dataset/Audio Dataset/Urban Sound/Torch/model.pth'

df_test = createAudioFrame(data_dir, classes)
myModel = AudioClassifier()
myModel.load_state_dict(torch.load(model_path))
myModel.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
testds = SoundDS(df_test,data_path)
val_dl = torch.utils.data.DataLoader(testds, batch_size=16, shuffle=False)
inference(myModel, val_dl,device)