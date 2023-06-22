import torch
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize
from model import EnhancedDetailNet
from PIL import Image

num_classes = 15
input_channels = 3

# Load the image
image_path = "./data/PlantVillage/Potato___Late_blight/5ec337d4-aff8-4174-bae6-72c246db3ce0___RS_LB 3175.JPG"
image = Image.open(image_path)

# Preprocess the image
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
preprocessed_image = transform(image).unsqueeze(0)

# Load the trained model
model = EnhancedDetailNet(num_classes=num_classes, input_channels=input_channels)
model.load_state_dict(torch.load("model_epoch100.pth"))
model.eval()

# Pass the image through the model
with torch.no_grad():
    output = model(preprocessed_image)

# Interpret the model output
probabilities = torch.softmax(output, dim=1)
predicted_class = torch.argmax(probabilities, dim=1)

# Convert predicted class to class name
class_names = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
               "Potato___healthy", "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
               "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
               "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus", "Tomato_healthy"]
predicted_class_name = class_names[predicted_class.item()]

# Print the predicted class
print("Predicted class:", predicted_class.item())
print("Predicted class name:", predicted_class_name)
