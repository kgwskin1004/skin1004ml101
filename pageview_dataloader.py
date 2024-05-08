import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


class PageViewsDataset(Dataset):
    def __init__(self, data_name="ampoule", sequence_length=5):
        if data_name == "ampoule":
            csv_file = "./data/pageview_ampoule.csv"
        elif data_name == "sun":
            csv_file = "./data/pageview_airfitsunserum.csv"
        elif data_name == "oil":
            csv_file = "./data/pageview_cleansingoil.csv"
        else:
            raise ValueError(
                'Invalid data_name specified. Please specify either "ampoule", "sun", or "oil"'
            )

        self.sequence_length = sequence_length
        self.data = pd.read_csv(csv_file, usecols=["Page_Views_Total"])
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data_normalized = self.scaler.fit_transform(self.data.values)

        self.x = []
        self.y = []

        for i in range(len(self.data_normalized) - sequence_length):
            self.x.append(self.data_normalized[i : i + sequence_length])
            self.y.append(self.data_normalized[i + sequence_length])

        self.x = torch.tensor(
            self.x, dtype=torch.float32
        )  # Shape: [n_samples, sequence_length, 1]
        self.y = torch.tensor(self.y, dtype=torch.float32).squeeze(
            1
        )  # Shape: [n_samples]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def inverse_transform(self, scaled_data):
        return self.scaler.inverse_transform(scaled_data)


if __name__ == "__main__":
    page_views_dataset = PageViewsDataset(data_name="ampoule")

    # Creating the DataLoader for batch processing
    dataloader = DataLoader(page_views_dataset, batch_size=4, num_workers=2)

    # Example of iterating through data
    for i, batch in enumerate(dataloader):
        print(batch)
        if i == 1:  # just to break the loop after 2 batches for demonstration
            break
