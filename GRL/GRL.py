import glob
import random
import logging
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

'''
This code has been extracted from adageo-WACV2021 GitHub Repository:
https://github.com/valeriopaolicelli/adageo-WACV2021
'''

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # during the forward pass, GRL acts as an identity transform (it just copy and paste data without changing anything)
        return x.clone()
    @staticmethod
    def backward(ctx, grads):
        # during the backward pass, GRL gets gradient of the subsequent level, 
        # multiplies it by -lambda (trade-off between Ly and Ld) and passes it to the precedent level
        dx = -grads.new_tensor(1) * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.shape[0], -1)
        return GradientReversalFunction.apply(x)


def get_discriminator(input_dim, num_classes=2):
    discriminator = nn.Sequential(
        GradientReversal(),
        nn.Linear(input_dim, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, num_classes)
    )
    return discriminator

grl_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class GrlDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root, datasets_paths, length=1000000):
        """
        datasets_paths is a list containing the folders which contain the N datasets.
        __len__() returns 1000000, and __getitem__(index) returns a random
        image, from dataset index % N, to ensure that each dataset has the 
        same chance of being picked
        """
        super().__init__()
        self.num_classes = len(datasets_paths)
        logging.info(f"GrlDataset has {self.num_classes} classes")
        self.images_paths = []
        for dataset_path in datasets_paths:
            self.images_paths.append(sorted(glob.glob(f"{dataset_root}/{dataset_path}/**/*.jpg", recursive=True)))
            logging.info(f"    Class {dataset_path} has {len(self.images_paths[-1])} images")
            if len(self.images_paths[-1]) == 0:
                raise Exception(f"Class {dataset_path} has 0 images, that's a problem!!!")
        self.transform = grl_transform
        self.length = length
    def __getitem__(self, index):
        num_class = index % self.num_classes
        images_of_class = self.images_paths[num_class]
        # choose a random one
        image_path = random.choice(images_of_class)
        tensor = self.transform(Image.open(image_path).convert("RGB"))
        return tensor, num_class
    def __len__(self):
        return self.length

