import torch
from torch.utils.data import Dataset

class WriterIDDataset(Dataset):
    def __init__(self, writer_ids):
        self.writer_ids = writer_ids  # list of writer IDs

    def __len__(self):
        return len(self.writer_ids)

    def __getitem__(self, idx):
        writer_id = self.writer_ids[idx]
        # Optionally, load content/text sample here
        # content = ...
        return {"writer_id": writer_id}
