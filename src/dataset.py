from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import TRAIN_DIR, VAL_DIR, IMAGE_SIZE, BATCH_SIZE

train_transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
val_transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor()
])

train_dataset=datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)
val_dataset=datasets.ImageFolder(
    root=VAL_DIR,
    transform=val_transform
)
train_loader=DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_loader=DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

def get_dataloaders():
    return train_loader, val_loader, train_dataset, val_dataset

if __name__ == "__main__":
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders()
    print(train_dataset.classes)
    print(train_dataset.class_to_idx)
    print(len(train_dataset))
    print(len(val_dataset))

    for images,labels in train_loader:
        print(images.shape)
        print(labels.shape)
        break