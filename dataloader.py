import os
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class ParquetDataset(Dataset):
    def __init__(self, file_paths, id):
        self.file_path = file_paths[id]
        self.parquet_file = pq.ParquetFile(self.file_path)
        self.table =  self.parquet_file.read().to_pandas().values

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        return self.table[idx][0]

dataset_path = "./wmt14/fr-en"
file_paths = os.listdir(dataset_path)
train_file_paths = []
test_file_paths = []
validation_file_paths = []
for file_path in file_paths:
    if file_path.startswith("train"):
        train_file_paths.append(os.path.join(dataset_path, file_path))
    elif file_path.startswith("test"):
        test_file_paths.append(os.path.join(dataset_path, file_path))
    elif file_path.startswith("validation"):
        validation_file_paths.append(os.path.join(dataset_path, file_path))

test_dataset = ParquetDataset(test_file_paths, 0)

def get_training_dataset(num, batch_size):
    train_dataset = ParquetDataset(train_file_paths, 0)
    for i in range(1, num):
        train_dataset = ConcatDataset([train_dataset, ParquetDataset(train_file_paths, i)]) 
    return DataLoader(train_dataset, batch_size, shuffle=True)

def get_test_dataset():
    return DataLoader(test_dataset, batch_size=1, shuffle=True)

def get_validation_dataset():
    validation_dataset = ParquetDataset(validation_file_paths, 0)
    return DataLoader(validation_dataset, batch_size=1, shuffle=True)
