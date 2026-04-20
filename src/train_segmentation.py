import torch
from torch.utils.data import DataLoader

from segmentation_model import UNet
from losses import combined_bce_dice_loss

def train_segmentation(model, dataset, epochs=10, batch_size=4, lr=1e-4, device="cuda"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for images, masks in loader:
            images = torch.tensor(images, dtype=torch.float32).to(device)
            masks = torch.tensor(masks, dtype=torch.float32).to(device)

            preds = model(images)
            loss = combined_bce_dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Segmentation Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    model = UNet()
    print("Segmentation training script ready.")