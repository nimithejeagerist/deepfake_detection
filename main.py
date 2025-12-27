import torch
import torch.nn as nn
import time
from dataset import DeepFakeDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from model import ResNet34
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize model
model = ResNet34(in_channels=3, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters())

# Training scheme
num_epochs = 100

def train(model, train_loader, criterion, optimizer, num_epochs, n_total_steps):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"[{epoch + 1}] loss: {running_loss / n_total_steps:.3f}")
    
    print("Finished Training")
    torch.save(model.state_dict(), "./resnet34_final.pt")

def evaluation(model, test_loader, ckpt):
    model.load_state_dict(torch.load(ckpt))
    model.to(device)
    model.eval()

    correct = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
    acc = 100.0 * correct / total
    print(f"Accuracy of the model: {acc:.2f}%")
    return acc

def eval_report(model, loader, ckpt):
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            p = model(x).argmax(1)
            ys.append(y.cpu()); ps.append(p.cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))
    
if __name__ == "__main__":
    start = time.time()
    
    n_total_steps = len(train_dataloader)
    train(model, train_dataloader, criterion, optimizer, num_epochs, n_total_steps)
    
    ckpt = "./resnet34_final.pt"
    fresh_model = ResNet34().to(device)
    evaluation(fresh_model, test_dataloader, ckpt)
    
    fresh_model = ResNet34().to(device)
    eval_report(fresh_model, test_dataloader, ckpt)

    end = time.time()
    print(f"Time: {(end-start)/60:.2f} min")
    