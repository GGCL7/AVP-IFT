
from typing import List, Tuple
import numpy as np
import torch
import torch.utils.data as Data

from peptide_encoding import feature_generator

def read_protein_sequences_from_fasta(file_path: str):

    sequences: List[str] = []
    labels: List[int] = []
    sequence = ''
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sequence:
                    sequences.append(sequence)
                    sequence = ''
                labels.append(1 if 'pos' in line else 0)
            else:
                sequence += line
        if sequence:
            sequences.append(sequence)
    return sequences, labels


def load_encoding_from_txt(file_path: str, max_rows: int = 50):

    sequences, labels = read_protein_sequences_from_fasta(file_path)
    encoded_sequences = feature_generator(file_path, max_rows=max_rows)
    encoded_sequences = np.asarray(encoded_sequences, dtype=np.float32)
    return encoded_sequences, labels


def load_features_from_txt(feature_file_path: str):

    features = np.loadtxt(feature_file_path)
    features = np.atleast_2d(features).astype(np.float32)
    return features


class PepDataset(Data.Dataset):

    def __init__(self, input_ids: np.ndarray, features: np.ndarray, labels: List[int]):
        assert len(input_ids) == len(features) == len(labels), \
            f"length：enc={len(input_ids)}, feat={len(features)}, label={len(labels)}"
        self.input_ids = input_ids
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.input_ids[idx], dtype=torch.float32),  # [L, F]
            torch.tensor(self.features[idx], dtype=torch.float32),   # [D]
            torch.tensor(self.labels[idx], dtype=torch.long)         # []
        )


def collate_fn(batch):

    xs, fs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    f = torch.stack(fs, dim=0)
    y = torch.stack(ys, dim=0)
    return x, f, y


def build_dataset(
    fasta_path: str,
    feature_txt_path: str,
    max_rows: int = 50
) -> Tuple[PepDataset, np.ndarray]:

    enc, labels = load_encoding_from_txt(fasta_path, max_rows=max_rows)  # (N, L, F)
    feats = load_features_from_txt(feature_txt_path)                     # (N, D)

    assert len(enc) == len(feats) == len(labels), \
        f"length：enc={len(enc)}, feats={len(feats)}, labels={len(labels)}"

    dataset = PepDataset(enc, feats, labels)
    return dataset, np.array(labels, dtype=np.int64)

