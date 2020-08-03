import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import copy
import os
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import cv2
import sys
from torch.autograd import Variable
import random

import statistics
from statistics import mode

from .model import resnet50

def classify(filepath):

  #Label file
  data_path = '/home/mcw/subha/DSC/dsc_django/UCF_Preprocess_Output/'
  classes = os.listdir(data_path)
  print(classes)
  decoder = {}
  id = list()


  for i in range(len(classes)):
      decoder[classes[i]] = i
  encoder = {}
  for i in range(len(classes)):
      encoder[i] = classes[i].split('_')[0]



  path = '/home/mcw/subha/DSC/dsc_django/UCF_Preprocess_Output/'

  for i in os.listdir(path):
    p1 = os.path.join(path,i)
    lis = os.listdir(p1)
    for j in lis:
      filename = filepath.split('_')[0]
      if filename in j:
        p2 = os.path.join(p1,j)
        id.append((i,p2))
        if len(id) == 10:
          break


  # print(id)
  print(len(id))
  # print(encoder)

  class video_dataset(Dataset):
      def __init__(self,frame_list,sequence_length = 16,transform = None):
          self.frame_list = frame_list
          self.transform = transform
          self.sequence_length = sequence_length
      def __len__(self):
          return len(self.frame_list)
      def __getitem__(self,idx):
          label,path = self.frame_list[idx]
          try:
            img = cv2.imread(path)
          except:
            print(path)
          seq_img = list()
          for i in range(16):
            
            img1 = img[:,128*i:128*(i+1),:]
            try:
              if(self.transform):
                img1 = self.transform(img1)
            except:
              print("Image Error: ",path)
              
            seq_img.append(img1)
          seq_image = torch.stack(seq_img)
          seq_image = seq_image.reshape(3,16,im_size,im_size)
          return seq_image,decoder[label]

  # Inference
  if True:
    model = resnet50(class_num=8).to('cuda')

    model.load_state_dict(torch.load('/home/mcw/subha/DSC/dsc_django/app/dl_scripts/Crime/c3d_19.h5'))

    print("Model Loaded")

    device = 'cuda'
    cls_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum = 0.9,weight_decay = 1e-4)


    im_size = 128
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    val_transforms = transforms.Compose([   transforms.ToPILImage(),
                                            transforms.Resize((im_size,im_size)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)
                                            ])

    val_data = video_dataset(id,sequence_length = 16,transform = val_transforms)
    val_loader = DataLoader(val_data,batch_size = 1,num_workers = 4)

    dataloaders = {'val':val_loader}

    prediction = []
    new = 1

    for batch_i, (X, y) in enumerate(dataloaders['val']):
      image_sequences = Variable(X.to(device), requires_grad=True)
      labels = Variable(y.to(device), requires_grad=False)
      predictions = model(image_sequences)
      new_prediction = predictions[0][0:5]
      new = torch.topk(new_prediction,4).values.detach().cpu()
      prediction.append(new.detach().max().item())
      print(new)
    # cnt = 0
    # for pred in prediction:
    #   if pred == 6:
    #     cnt +=1
    print(prediction)
    return 

# infer('home/mcw/subha/DeepSeeCrime/deepseecrime_django/static_cdn/media_root/input_videos/Abuse03_j.mp4')



