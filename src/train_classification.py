import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classification_model import CNNClassifier

def train_classification(model, dataset, epochs=10, batch_size=8, lr=1e-4, device="cuda"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for images, labels in loader:
            images = torch.tensor(images, dtype=torch.float32).to(device)
            labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Classification Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    model = CNNClassifier()
    print("Classification training script ready.")