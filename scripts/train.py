import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Placeholder: Define your model here
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Flatten input (224x224x3) to a vector of size 224*224*3 for the fully connected layer
        self.fc = nn.Linear(224 * 224 * 3, 10)  # Example dimensions (adjust for actual number of classes)

    def forward(self, x):
        # Flatten the input tensor from (batch_size, 3, 224, 224) to (batch_size, 224*224*3)
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Prepare the dataset and DataLoader
    train_dataset = datasets.ImageFolder(root="./data/processed/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model
    model = SimpleModel()

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the correct device

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the same device as the model

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()  # Track the running loss

        # Print the average loss for the epoch
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # Optionally, save the model after training
    torch.save(model.state_dict(), "./model.pt")

