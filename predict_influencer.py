import torch
from torch.utils.data import DataLoader
from influencer_dataloader import InfluencerDataset
from model import InfluenceModel

# Load the trained model
model = InfluenceModel()
model.load_state_dict(torch.load("./saved_model/influence_model.pth"))
model.eval()  # Set the model to evaluation mode

# Dataset and DataLoader
dataset = InfluencerDataset()
dataloader = DataLoader(
    dataset, batch_size=1, shuffle=False
)  # No need to shuffle for evaluation

# Dictionary to store influencer URLs and their predicted scores
influencer_scores = {}

# Evaluate the model
with torch.no_grad():  # Disable gradient computation
    for features, url in dataloader:
        # Features are [view, save, like, comment, share]
        # Assuming 'cost' is included in the features as the last item
        cost = features[:, -1]  # Extract cost
        inputs = features[:, :-1]

        # Forward pass to get outputs
        outputs = model(inputs, cost)
        influencer_scores[url[0]] = outputs.item()

# Sorting influencers by their scores in descending order
sorted_influencers = sorted(influencer_scores.items(), key=lambda x: x[1], reverse=True)

# Print or save the sorted list
for url, score in sorted_influencers:
    print(f"Influencer URL: {url}, Score: {score}")

# Optional: Save the sorted list to a file
with open("sorted_influencer_scores.txt", "w") as file:
    for url, score in sorted_influencers:
        file.write(f"Influencer URL: {url}, Score: {score}\n")
