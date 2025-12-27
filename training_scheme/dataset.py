import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import decode_image

class DeepFakeDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = None
        image = None
        label = int(self.img_labels.iloc[idx, 1])
        if label == 1:
            img_path = os.path.join(self.img_dir[0], self.img_labels.iloc[idx, 0])
        else:
            img_path = os.path.join(self.img_dir[1], self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label