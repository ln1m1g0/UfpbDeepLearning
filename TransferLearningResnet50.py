

# # Creating a dictionary of lists using list comprehension
# d = dict((val, range(int(val), int(val) + 2))
#                   for val in range(10))
 
# print(d)
# import pandas as pd


# def common(a, b):
#     return any(i in b for i in a)

# a = [1, 2, 3, 4, 5]
# b = [9, 8, 7, 6, 2]
# c = (set(a) & set(b))
# d = (a) == (b)
# # print(list(set(a) ^ set(b)))

# from numpy import random
# import numpy as geek

# array = ( 3, 4 , 7, 8, 0, 10, 23)
# print("INPUT ARRAY : \n", array) 
  
  
# # returning Indices of the min element 
# # as per the indices 
# print("\nIndices of min element : ", geek.argmin(array, axis=0)) 

import torch
import torchvision.models as TModels
from torchsummary import summary
import torch.nn as nn

ResNet50 = TModels.resnet50(weights=TModels.ResNet50_Weights.IMAGENET1K_V2)
summary(ResNet50, (3,224,224))

for param in ResNet50.parameters():
    param.requires_grad = False

ResNet50.fc = nn.Sequential(nn.Linear(2048, 256),
                         nn.ReLU(),
                         nn.Linear(256, 1),
                         nn.Sigmoid())

for param in ResNet50.fc.parameters():
    param.requires_grad = True