import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from influencer_dataloader import (
    InfluencerDataset,
)  # Ensure this module exists and is correctly implemented
from model import (
    InfluenceModel,
)  # Ensure this module exists and is correctly implemented
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.01
num_epochs = 1000
batch_size = 1

# Dataset and DataLoader
dataset = InfluencerDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, and Optimizer
model = InfluenceModel()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Initialize a list to store the losses
epoch_losses = []

# Training Loop
for epoch in range(num_epochs):
    predicted_score = {}
    for features, url in dataloader:
        # Features are [view, save, like, comment, share]
        # Assuming 'cost' is included in the features as the last item
        cost = features[:, -1]  # Extract cost
        inputs = features[:, :-1]

        # Forward pass
        outputs = model(inputs, cost)

        predicted_score[url[0]] = outputs

    # Compute the hinge loss
    loss = dataset.hinge_loss(predicted_score)
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss = loss.item()
    epoch_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "./saved_model/influence_model.pth")

# Plotting the loss
plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, label="Training Loss")
plt.title("Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
