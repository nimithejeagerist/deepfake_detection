import torch
from dataset import DeepFakeDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt


annotations_file = ["data/train.csv", "data/test.csv"]
img_dir = ["C:/Users/nimza/Documents/dd/fakes_cropped", "C:/Users/nimza/Documents/dd/real_cropped"]

# Define transforms for train and test
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

train_features, train_label = next(iter(train_dataloader))
print(f"Features batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_label.size()}")
img = train_features[0].squeeze()
img = img.permute(1, 2, 0)
label = train_label[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")