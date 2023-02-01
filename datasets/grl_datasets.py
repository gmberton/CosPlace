import os
import torch
import random
import logging
import numpy as np
from glob import glob
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True

def open_image(path):
    return Image.open(path).convert("RGB")
    
class GrlDataset(torch.utils.data.Dataset):
    def __init__(self, sf_xs_train_path, target_path, length = 1000000):
        """
        datasets_paths is a list containing the folders which contain the N datasets. (in our case 2 datasets containing source + pseudo, target)
        __len__() returns 1000000, and __getitem__(index) returns a random
        image, from dataset index % N, to ensure that each dataset has the 
        same chance of being picked
        """
        super().__init__()
        self.num_classes = 3
        logging.info(f"GrlDataset has {self.num_classes} domain classes")
        self.images_paths = []
        #for each dataset path.. for ex. /content/small/train (source + pseudo), /content/AML23-CosPlace/our_FDA/target

        #Add the 5 target images paths to array
        self.images_paths.append(glob(f"{target_path}/**/*.jpg", recursive=True))

        source_paths = []
        pseudo_paths = []

        #Add source and pseudo-target images paths to array
        for root, dirs, files in os.walk(sf_xs_train_path, topdown=True):
            for i in range(0, len(files), 1): 
                if '.DS_Store' in files[i]:
                    continue

                if "NIGHT" in files[i]:
                    pseudo_paths.append(os.path.join(root, files[i]))
                else:
                    source_paths.append(os.path.join(root, files[i]))

        self.images_paths.append(source_paths)
        self.images_paths.append(source_paths)
                
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((512,512)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.length = length


    def __getitem__(self, index):
        num_class = index % self.num_classes
        images_of_class = self.images_paths[num_class]
        # choose a random one
        image_path = random.choice(images_of_class)
        tensor = self.transform(Image.open(image_path).convert("RGB"))

        #If the image name contains "NIGHT" --> domain = 1 (night), else domain = 0 (not night)
        domain = 1 if "NIGHT" in os.path.basename(image_path) else 0

        return tensor, domain


    def __len__(self):
        return self.length