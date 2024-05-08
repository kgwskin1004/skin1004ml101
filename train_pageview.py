import torch.optim as optim
from model import LSTM
import torch.nn as nn
from pageview_dataloader import PageViewsDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

# Initialization
model = LSTM()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 100

# DataLoader
train_data_loaders = [
    DataLoader(
        PageViewsDataset(data_name="ampoule", sequence_length=14),
        batch_size=1,
        shuffle=False,
    ),
    DataLoader(
        PageViewsDataset(data_name="sun", sequence_length=14),
        batch_size=1,
        shuffle=False,
    ),
]
test_data_loader = DataLoader(
    PageViewsDataset(data_name="oil", sequence_length=14), batch_size=1, shuffle=False
)

# To store losses for plotting
train_losses = []
eval_losses = []

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_losses = []
    for train_data_loader in train_data_loaders:
        for inputs, labels in train_data_loader:
            optimizer.zero_grad()
            model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size),
            )

            y_pred = model(inputs)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

            epoch_losses.append(single_loss.item())

    train_loss_avg = sum(epoch_losses) / len(epoch_losses)
    train_losses.append(train_loss_avg)
    print(f"Epoch {epoch + 1} Training Loss: {train_loss_avg}")

    # Evaluation
    model.eval()
    test_losses = []
    for inputs, labels in test_data_loader:
        with torch.no_grad():
            model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size),
            )
            y_pred = model(inputs)
            test_loss = loss_function(y_pred, labels)
            test_losses.append(test_loss.item())

    test_loss_avg = sum(test_losses) / len(test_losses)
    eval_losses.append(test_loss_avg)
    print(f"Epoch {epoch + 1} Evaluation Loss: {test_loss_avg}")

# Save the model
torch.save(model.state_dict(), "./saved_model/lstm_model.pth")

# Plotting the losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(eval_losses, label="Evaluation Loss")
plt.title("Training and Evaluation Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
