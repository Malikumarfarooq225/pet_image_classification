import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io

# Define the model class (same as training)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(224 * 224 * 3, 10)  # Example dimensions

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        return self.fc(x)

# Load the trained model
def load_model():
    model = SimpleModel()
    model.load_state_dict(torch.load("./model.pt"))
    model.eval()
    return model

# Preprocess the input image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Main function for the Streamlit app
def main():
    st.title("Image Classification with PyTorch")
    
    st.write("Upload an image to classify it using the trained model.")
    
    # Upload Image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Preprocess the image and run the model
        model = load_model()
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)
            

        # Display prediction
        st.write(f"Predicted Class: {predicted_class.item()}")

if __name__ == "__main__":
    main()
