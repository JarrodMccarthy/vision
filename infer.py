import torch
import torch.optim as optim
import torch.nn as nn
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

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")

from typing import Any


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
}


class AlexNet1(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.softmax(x, dim = -1)
        return x

def alexnet1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet1:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet1(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

class faceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        # """
        # Args:
        #     csv_file (string): Path to the csv file with annotations.
        #     root_dir (string): Directory with all the images.
        #     transform (callable, optional): Optional transform to be applied
        #         on a sample.
        # """
        self.label_name_dict = {
            "yes": 1,
            "no": 0
        }
        self.df = pd.read_csv(csv_file, names=["name", "face"])
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = str(idx) +'.jpg'

        if idx < 500:
            path = os.path.join(os.getcwd(), 'data', 'faces', 'yes', img_name)
        else:
            path = os.path.join(os.getcwd(), 'data', 'faces', 'no', img_name)

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image/255
        label = self.df.iloc[idx, 1:]
        
        label = int(label)
        sample = {'image': image, 'label': label}
        #print(sample)
        if self.transform:
            sample = self.transform(sample)
        #print(img_name, label)
        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size = 256):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = cv2.resize(image, (self.output_size, self.output_size))
        
        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        #print(np.shape(image))
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        #image = image.transpose((2, 0, 1))
        
        return {'image': image,
                'label': torch.tensor(label)}

def showstuff(images, outputs):
    fig = plt.figure(figsize=(8,8))
    rows = 2
    cols = 2
    
    for i in range(1, rows*cols+1):
        fig.add_subplot(rows, cols, i)
        outs = outputs.tolist()
        if i == 1 or i == 2:
            if i == 1:
                plt.title("No Face")
            elif i == 2:
                plt.title("Face")
            image = images[i-1].detach().cpu().numpy()
            image = image.transpose((1, 2, 0))
            plt.imshow(image, cmap='gray')
        elif i == 3:
            plt.bar([0,1], outs[0])
            plt.xticks([0,1])
        elif i == 4:
            plt.bar([0,1], outs[1])
            plt.xticks([0,1])
    plt.show()


root = os.getcwd()
path = os.path.join(root, 'data', 'faces', 'faces.csv')

data = faceDataset(path, transform=transforms.Compose([Rescale(256),ToTensor()]))

ver_loader = torch.utils.data.DataLoader(
    data,
    batch_size=2,
    shuffle=True
    #num_workers=4
)
#saveimagepath = os.path.join('data','faces','unclassified')


model = alexnet1(pretrained=False)

model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)

BEST_MODEL_PATH = os.path.join(os.getcwd(),'bestmodels','faces','best_model.pth')

model.load_state_dict(torch.load(BEST_MODEL_PATH))

model = model.float()
model.eval()

# for name, param in model.named_parameters():
#     if "classifier.6" in name:
#         print("{}: {}".format(name, param))

for i, item in enumerate(ver_loader):
    images = item['image'].to(device)
    labels = item['label'].to(device)
    outputs = model(images.float())
    print("Outputs: {}".format(outputs))
    print("Labels: {}".format(labels))
    print("Outputs: {}".format(outputs.argmax(1)))
    showstuff(images, outputs)


    break
