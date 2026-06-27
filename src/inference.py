import torch
from model import create_model
from torchvision import transforms
from PIL import Image
from config import IMAGE_SIZE,MODEL_PATH

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=create_model()
model.load_state_dict(
    torch.load(MODEL_PATH,map_location=device)
)
model=model.to(device)
model.eval()

transform=transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor()
])
classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]
image_path="dataset/validation/images/crazing/crazing_247.jpg"
image=Image.open(image_path).convert("RGB")

image=transform(image)
image=image.unsqueeze(0)
image=image.to(device)

with torch.no_grad():
    outputs=model(image)

probabilities = torch.softmax(outputs, dim=1)

confidence, predicted_class = torch.max(probabilities, dim=1)

predicted_label = classes[predicted_class.item()]
confidence_percent = confidence.item() * 100

print("Prediction:", predicted_label)
print(f"Confidence: {confidence_percent:.2f}%")






# Image.open()        -> resmi açar
# convert("RGB")      -> 3 kanal yapar
# transform(image)    -> 224x224 + tensor
# unsqueeze(0)        -> batch boyutu ekler
# model(image)        -> tahmin üretir
# softmax             -> olasılığa çevirir
# torch.max           -> en yüksek olasılığı bulur