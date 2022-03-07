import numpy as np
from PIL import Image
import torch.utils.data as data
import os
import torch

class ImageDataset(data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image
    
def convert_dimensions(image):
    # input form (NUM, CHANNEL, HEIGHT, WIDTH)
    # returns (NUM, HEIGHT, WIDTH, CHANNEL)
    
    image = np.moveaxis(image,1,3)
    return image