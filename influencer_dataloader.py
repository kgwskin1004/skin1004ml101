import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class InfluencerDataset(Dataset):
    def __init__(self):
        csv_file = "./data/influencer_data.csv"
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        # Load data
        self.influencers_data = pd.read_csv(csv_file)

        # Clean and convert data
        # Remove backslashes and commas, then convert to appropriate types
        self.influencers_data["num_follower"] = (
            self.influencers_data["num_follower"].str.replace(",", "").astype(int)
        )
        self.influencers_data["cost"] = (
            self.influencers_data["cost"]
            .str.replace(r"\\", "")
            .str.replace(",", "")
            .astype(float)
        )
        self.influencers_data["view"] = (
            self.influencers_data["view"].str.replace(",", "").astype(int)
        )
        self.influencers_data["save"] = (
            self.influencers_data["save"].str.replace(",", "").astype(int)
        )
        self.influencers_data["like"] = (
            self.influencers_data["like"].str.replace(",", "").astype(float)
        )
        self.influencers_data["comment"] = (
            self.influencers_data["comment"].str.replace(",", "").astype(float)
        )
        self.influencers_data["share"] = (
            self.influencers_data["share"].str.replace(",", "").astype(float)
        )

    def __len__(self):
        return len(self.influencers_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.influencers_data.iloc[idx]
        features = torch.tensor(
            [
                data["view"],
                data["save"],
                data["like"],
                data["comment"],
                data["share"],
                data["cost"],
            ]
        )
        influencer_url = data["Influencer_profile_url"]

        return features, influencer_url

    def hinge_loss(self, predicted_score, margin=1.0):
        """
        For every pair of influencer url in the self.influencers_data, the hinge loss is computed as:
        max(0, margin - (predicted_score_lower_influencer - predicted_score_higher_influencer))
        """

        # iterate through the dataset to compute the hinge loss
        loss = 0
        for i in range(len(self.influencers_data)):
            for j in range(i + 1, len(self.influencers_data)):
                i_url = self.influencers_data.iloc[i]["Influencer_profile_url"]
                j_url = self.influencers_data.iloc[j]["Influencer_profile_url"]
                # Get the predicted scores for the two influencers
                score_i = predicted_score[i_url]
                score_j = predicted_score[j_url]

                # Compute the hinge loss
                loss += max(0, margin - (score_j - score_i))

        return loss


if __name__ == "__main__":
    # Example usage
    influencer_dataset = InfluencerDataset()

    # Creating the DataLoader for batch processing
    dataloader = DataLoader(
        influencer_dataset, batch_size=4, shuffle=True, num_workers=2
    )

    # Example of iterating through data
    for i, batch in enumerate(dataloader):
        print(batch)
        if i == 1:  # just to break the loop after 2 batches for demonstration
            break
