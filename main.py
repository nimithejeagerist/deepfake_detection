import os
import torch
import torch.nn as nn
import time
import numpy as np
from dataset import DeepFakeDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from model import ResNet34Original
from torch.utils.data import random_split
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
annotations_file = ["data/train.csv", "data/test.csv"]
img_dir = ["C:/Users/nimza/Documents/dd/fakes_cropped", "C:/Users/nimza/Documents/dd/real_cropped"]

num_workers = min(8, os.cpu_count() or 4)
pin_memory = (device.type == "cuda")

transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])

train_data = DeepFakeDataset(annotations_file=annotations_file[0], img_dir=img_dir, transform=transforms)
test_data = DeepFakeDataset(annotations_file=annotations_file[1], img_dir=img_dir, transform=transforms)

batch_size = 64
val_frac = 0.1
n = len(train_data)
n_val = int(n * val_frac)
n_train = n - n_val

train_subset, val_subset = random_split(
    train_data,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_dataloader = DataLoader(
    train_subset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=pin_memory
)
val_dataloader = DataLoader(
    val_subset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=pin_memory
)
test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
)

model = ResNet34Original(in_channels=3, num_classes=2).to(device)
# get labels from the underlying dataset using the indices in the subset
train_indices = train_subset.indices
y_train = train_data.img_labels.iloc[train_indices, 1].astype(int).to_numpy()

counts = np.bincount(y_train, minlength=2)   
weights = torch.tensor([1.0 / counts[0], 1.0 / counts[1]], dtype=torch.float32, device=device)
weights = weights / weights.sum() * 2.0     

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.SGD(model.parameters())

num_epochs = 50
BEST_CKPT_PATH = "./resnet34_best.pt"

def balanced_accuracy(model, loader):
    model.eval()
    cm = torch.zeros(2, 2, dtype=torch.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            p = model(x).argmax(1)
            for t, pr in zip(y.view(-1), p.view(-1)):
                cm[t.long(), pr.long()] += 1

    # recalls: TP / (TP+FN) per class
    recall0 = cm[0,0].float() / (cm[0,0] + cm[0,1]).clamp(min=1)
    recall1 = cm[1,1].float() / (cm[1,1] + cm[1,0]).clamp(min=1)
    return float((recall0 + recall1) / 2.0)



def train(model, train_loader, criterion, optimizer, num_epochs):
    best_score = -1.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        score = balanced_accuracy(model, val_dataloader)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            tag = " (best saved)"
        else:
            tag = ""

        print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f} | val_bal_acc={score:.3f} | best={best_score:.3f}{tag}")


    print("Finished Training")
    torch.save(model.state_dict(), "./resnet34_final.pt")


def eval_report(model, loader, ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            p = model(x).argmax(1)
            ys.append(y.cpu())
            ps.append(p.cpu())

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))


if __name__ == "__main__":
    start = time.time()

    train(model, train_dataloader, criterion, optimizer, num_epochs)

    best_model = ResNet34Original(in_channels=3, num_classes=2).to(device)
    eval_report(best_model, test_dataloader, BEST_CKPT_PATH)

    end = time.time()
    print(f"Time: {(end-start)/60:.2f} min")
