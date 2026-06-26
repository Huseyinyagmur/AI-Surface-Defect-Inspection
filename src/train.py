import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import create_model
from config import LEARNING_RATE, NUM_EPOCHS, MODEL_PATH


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

train_loader, val_loader, train_dataset, val_dataset = get_dataloaders()

model = create_model()
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_accuracy = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

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

    epoch_loss = running_loss / len(train_loader)
    print(f"Train Loss: {epoch_loss:.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += len(labels)

    val_accuracy = correct / total
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), MODEL_PATH)
        print("Best model saved.")