import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_dataloaders
from model import create_model
from config import MODEL_PATH

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=create_model()
model.load_state_dict(
    torch.load(MODEL_PATH,map_location=device)
)
model=model.to(device)
model.eval()

_,val_loader,_,val_dataset = get_dataloaders()

all_labels=[]
all_predictions=[]

with torch.no_grad():
    for images,labels in val_loader:
        images=images.to(device)
        labels=labels.to(device)

        outputs=model(images)
        prediction=torch.argmax(outputs,dim=1)

        all_predictions.extend(prediction.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
classes=val_dataset.classes
cm=confusion_matrix(all_labels,all_predictions)
report=classification_report(all_labels,all_predictions,target_names=classes)

print(cm)
print(report)