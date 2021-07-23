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

import torch
import torch.nn as nn
import torch.nn.init as init
#from .utils import load_state_dict_from_url
from typing import Any

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
}


class AlexNet1(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2), #AlexNet -> nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
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
            nn.Linear(4096, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)


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

def plotter(accuracy, error):
    plt.figure()
    plt.scatter(np.linspace(0,len(accuracy)-1, len(accuracy)), accuracy, label='Accuracy')
    plt.legend()
    plt.show()
    #plt.hold(True)
    plt.figure()
    plt.scatter(np.linspace(0,len(error)-1, len(error)), error, label='Error')
    plt.legend()
    plt.show()
    return 

def getonehot(labels):
    if labels.item() == 1:
        label = torch.tensor([[0.0, 1.0]])
        return label
    elif labels.item() == 0:
        label = torch.tensor([[1.0, 0.0]])
        return label
    else:
        print("Failed to get One hot for : {}".format(labels))
        return None

root = os.getcwd()
path = os.path.join(root, 'data', 'faces', 'faces.csv')
verpath = os.path.join(root, 'data', 'faces', 'facesunclass.csv')

verdata = faceDataset(verpath, transform=transforms.Compose([Rescale(256),ToTensor()]))

ver_loader = torch.utils.data.DataLoader(
    verdata,
    batch_size=1,
    shuffle=True
    #num_workers=4
)

data = faceDataset(path, transform=transforms.Compose([Rescale(256),ToTensor()]))

train_dataset, test_dataset = torch.utils.data.random_split(data, [len(data) - 250, 250])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
    #num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=True
    #num_workers=4
)

model = alexnet1(pretrained=False, num_classes=2)

model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)

# BEST_MODEL_PATH = os.path.join(root,'bestmodels','faces','best_model.pth')
# model.load_state_dict(torch.load(BEST_MODEL_PATH))

model = model.float()

# for name, param in model.named_parameters():
#     print("{}: {}".format(name, param.size()))

NUM_EPOCHS = 10
BEST_MODEL_PATH = os.path.join('bestmodels','faces','best_model.pth')
best_accuracy = 0.0

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0)

accuracy = list()
testerror = list()
trainerror = list()
for epoch in range(NUM_EPOCHS):
    
    print("Starting Train Loader...")
    model.train()
    train_error_count = 0.0
    for i, item in enumerate(train_loader):
        images = item['image'].to(device)
        labels = item['label'].to(device)
        probs = getonehot(labels).to(device)
        optimizer.zero_grad()
        outputs = model(images.float())
        #print(outputs, probs)
        loss = F.binary_cross_entropy(outputs, probs)
        loss.backward()
        optimizer.step()
        train_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
        #print("Train Summary {}: Correct: {}".format(i, labels.item()==outputs.argmax(1).item()))
    
    print("Starting Test Loader...")
    model.eval()
    test_error_count = 0.0
    for i, item in enumerate(test_loader):
        images = item['image'].to(device)
        labels = item['label'].to(device)
        outputs = model(images.float())
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
        #print("Test Summary {}: Correct: {}".format(i, labels.item()==outputs.argmax(1).item()))
    
    test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
    train_accuracy = 1.0 - float(train_error_count) / float(len(test_dataset))
    accuracy.append(test_accuracy)
    trainerror.append(train_error_count)
    testerror.append(test_error_count)

    

    print('Epoch {}: Test Accuracy: {:.3f}, Train Accuracy: {:.3f}'.format(epoch, test_accuracy, train_accuracy))
    if test_accuracy >= best_accuracy:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_accuracy = test_accuracy

plotter(test_accuracy, testerror)
plotter(train_accuracy, testerror)
# for i, item in enumerate(ver_loader):
#     images = item['image'].to(device)
#     labels = item['label'].to(device)
#     outputs = model(images.float())
#     print("Outputs: {}".format(outputs))
#     print("Labels: {}".format(labels))
#     print("Outputs: {}".format(outputs.argmax(1)))
#     for image in images:
#         #label = model(image.float())
#         #print("Label: {}".format(label))
#         image = image.detach().cpu().numpy()
#         #print(type(image))
#         image = image.transpose((1, 2, 0))
#         #print(np.shape(image))
#         plt.figure()
#         plt.imshow(image)
#     plt.show()
#     break