from model import LSTM
import torch.nn as nn
import torch
from pageview_dataloader import PageViewsDataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np

test_data_loader = DataLoader(
    PageViewsDataset("oil", sequence_length=14), batch_size=1, shuffle=False
)
predictions = []
true_values = []

loss_function = nn.MSELoss()

model = LSTM()
# Evaluation Setup
model.load_state_dict(torch.load("./saved_model/lstm_model.pth"))
model.eval()


for inputs, labels in test_data_loader:
    with torch.no_grad():
        model.hidden_cell = (
            torch.zeros(1, 1, model.hidden_layer_size),
            torch.zeros(1, 1, model.hidden_layer_size),
        )
        y_pred = model(inputs)
        predictions.extend(y_pred.numpy())  # Append batch predictions
        true_values.extend(labels.numpy())  # Append true labels

# Convert lists to numpy arrays
predictions = np.array(predictions)
true_values = np.array(true_values)

# Inverse transform the predictions and true values
dataset = PageViewsDataset(
    "oil", sequence_length=5
)  # Reinitialize dataset to access scaler
predictions_original = dataset.inverse_transform(predictions.reshape(-1, 1)).flatten()
true_values_original = dataset.inverse_transform(true_values.reshape(-1, 1)).flatten()

plt.figure(figsize=(10, 5))
plt.plot(true_values_original, label="True Values")
plt.plot(predictions_original, label="Predictions")
plt.title("Comparison of True Values and Predictions")
plt.xlabel("Time")
plt.ylabel("Page Views")
plt.legend()
plt.show()
