import numpy as np
import cv2 as cv
import torch
from pathlib import Path
from torch.utils.data import Dataset

class ClothingDataset(Dataset):
    def __init__(self, images, label_encoder, transform=None):
        self.images = images

        self.label_encoder = label_encoder
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        input_path = self.images[idx]
        input_name = input_path.stem
        input_label = input_path.parent.parent / Path('labels') / Path(input_name + '.npy')
        
        input = cv.imread(str(input_path))


        if self.transform:
            input = self.transform(input)

        label = torch.tensor(
            self.label_encoder.encode(np.load(input_label))
        )

        return input, label