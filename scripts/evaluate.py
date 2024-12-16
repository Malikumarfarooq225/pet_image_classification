import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the model class again (same as in training)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(224 * 224 * 3, 10)  # Example dimensions

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        return self.fc(x)

if __name__ == "__main__":
    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved model weights
    model = SimpleModel()  # Initialize the model architecture
    model.load_state_dict(torch.load("./model.pt"))  # Load the saved state dict
    model.to(device)  # Move the model to the selected device (GPU or CPU)
    model.eval()  # Set the model to evaluation mode

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare the dataset and DataLoader
    val_dataset = datasets.ImageFolder(root="./data/processed/val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    # Evaluation loop
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the same device as the model
            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            total += labels.size(0)  # Update total samples count
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Print the accuracy
    print(f"Accuracy: {100 * correct / total}%")
