import torch
from dataset import DeepFakeDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2

annotations_file = ["data/train.csv", "data/test.csv"]
img_dir = ["C:/Users/nimza/Documents/dd/fakes", "C:/Users/nimza/Documents/dd/real"]

transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])

train_data = DeepFakeDataset(annotations_file=annotations_file[0], img_dir=img_dir, transform=transforms)
test_data = DeepFakeDataset(annotations_file=annotations_file[1], img_dir=img_dir, transform=transforms)

batch_size = 64

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y {y.shape} {y.dtype}")
    break