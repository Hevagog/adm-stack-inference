import torch
import h5py
import numpy as np
from torch.utils.data import Dataset


class LazyDSFDataset(Dataset):
    def __init__(
        self,
        h5_path,
        num_tags_list,
        binary_labels,
    ):
        """
        Args:
            h5_path (str): Path to the generated .h5 file.
            num_tags_list (list): Normalized list of number of tags for each sample.
            binary_labels (np.array): Result of MultiLabelBinarizer on the tags.
        """
        self.h5_path = h5_path
        self.num_tags_list = num_tags_list
        self.binary_labels = binary_labels
        self.archive = None

        with h5py.File(h5_path, "r") as f:
            self.length = len(f["question_ids"])

    def _init_archive(self):
        if self.archive is None:
            self.archive = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._init_archive()
        body_seq = torch.from_numpy(self.archive["body_seq"][idx]).float()

        # Saved as: True=Real Word, False=Padding
        # Inverting the logical state because nn.MultiheadAttention expects: True=Padding (Ignore), False=Real Word
        saved_mask = self.archive["body_mask"][idx]
        padding_mask = torch.from_numpy(~saved_mask)

        title_emb = torch.from_numpy(self.archive["title_emb"][idx]).float()
        n_tags = torch.tensor(self.num_tags_list[idx], dtype=torch.float32)
        label_vector = torch.tensor(self.binary_labels[idx], dtype=torch.float32)

        return title_emb, body_seq, padding_mask, n_tags, label_vector
