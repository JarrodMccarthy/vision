import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import cv2

import math
import os

# label_name_dict = {
#             "non-motorized_vehicle":0,
#             "articulated_truck":1,
#             "background":2,
#             "bicycle":3,
#             "bus":4,
#             "car":5,
#             "motorcycle":6,
#             "pedestrian":7,
#             "pickup_truck":8,
#             "single_unit_truck":9,
#             "work_van":10
#         }

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")

# model = models.alexnet(pretrained=False)
# model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)



# BEST_MODEL_PATH = os.path.join('bestmodels','faces','best_model.pth')
# model.load_state_dict(torch.load(BEST_MODEL_PATH))
# model = model.float()
# model.eval()



saveimagepath = os.path.join('data','faces','article')
# define a video capture object
vid = cv2.VideoCapture(0)

i = 0
while(True):
    # Capture the video frame
    # by frame
    ret, image = vid.read()
    cv2.imshow('Image',image)

    #image1 = cv2.resize(image1, (256,256))
    # imagename = os.path.join(saveimagepath, str(i)+'.jpg')
    # cv2.imwrite(imagename, image1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()