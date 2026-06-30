import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os

if not os.path.exists('classes.json'):
    raise FileNotFoundError("No se encontró classes.json. Debes entrenar el modelo primero.")

with open('classes.json', 'r') as f:
    classes = json.load(f)
def load_model():
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    
    # Cargar los pesos genéricos
    model.load_state_dict(torch.load('custom_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
        
    input_tensor = transform(image).unsqueeze(0) 
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    return {classes[i]: float(probabilities[i]) for i in range(len(classes))}

#Interfaz Gradio
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", sources=["upload", "webcam"], label="Sube una imagen para clasificar"),
    outputs=gr.Label(num_top_classes=3, label="Predicción"),
    title="Clasificador de Imágenes Personalizado (ResNet18)",
    description=f"Este modelo ha sido entrenado para clasificar {len(classes)} categorías distintas: {', '.join(classes)}."
)

if __name__ == "__main__":
    interface.launch()