import torch
from torch.utils.data import Dataset
import numpy as np


class StackOverflowDataset(Dataset):
    def __init__(
        self,
        title_embeddings,
        body_embeddings,
        num_tags_list,
        binary_labels,
        train=True,
        test_size=0.15,
        seed=42,
    ):
        """
        Args:
            title_embeddings (np.array): Precomputed title embeddings.
            body_embeddings (np.array): Precomputed body embeddings.
            num_tags_list (list): List of number of tags for each sample.
            binary_labels (np.array): Result of MultiLabelBinarizer on the tags.
            train (bool): Indicates if the dataset is for training or not.
            test_size (float): Proportion of the dataset to include in the test split.
            seed (int): Random seed for reproducibility.
        """
        total_len = len(title_embeddings)
        indices = np.arange(total_len)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        test_size = int(test_size * total_len)

        if train:
            selected_indices = indices[:-test_size]
        else:
            selected_indices = indices[-test_size:]

        self.title_embs = title_embeddings[selected_indices]
        self.body_embs = body_embeddings[selected_indices]
        num_tags_list = np.array(num_tags_list)
        self.num_tags = num_tags_list[selected_indices]
        self.labels = binary_labels[selected_indices]
        self.train = train

    def __len__(self):
        return len(self.title_embs)

    def __getitem__(self, idx):
        t_emb = torch.tensor(self.title_embs[idx], dtype=torch.float32)
        b_emb = torch.tensor(self.body_embs[idx], dtype=torch.float32)
        # scale num_tags to [0, 1] range. StackOverflow allows up to 5 tags.
        n_tag = torch.tensor([self.num_tags[idx] / 5.0], dtype=torch.float32)
        label_vector = torch.tensor(self.labels[idx], dtype=torch.float32)

        return t_emb, b_emb, n_tag, label_vector
