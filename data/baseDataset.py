from torch.utils.data import Dataset, DataLoader
import glob
from utilities.utils import load_json
import numpy as np
import torch

class BaseDataset(Dataset):
    """Text dataset from json"""

    def __init__(self, labels_path, category=None, transform=None, target_transform=None, only=["key", "value"], unknown_drop_rate=0.):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.only = only
        self.unknown_drop_rate = unknown_drop_rate
        self.data, self.category = self.load_data(labels_path, category)
        
        self.transform = transform
        self.target_transform = target_transform
        
    
    def fetch_data(self, labels_path, category=None):
        list_label = glob.glob(labels_path + "/*")
        data = []

        for label_path in list_label:
            CASIA_output = load_json(label_path, key_list=category, only=self.only)

            # fetch label item
            items = [(item.get("text"), item.get("type"), item.get("key_type"), label_path) for item in CASIA_output]
            data.extend(items)
        
        return data

    def load_data(self, labels_path, category=None):
        if isinstance(labels_path, list):
            data = []
            for label_path in labels_path:
                data.extend(self.fetch_data(label_path, category=category))
        else:
            data = self.fetch_data(labels_path, category=category)

        category = list(np.unique([item[1] for item in data]))
        
        return data, category
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
      
        text, cat = self.data[idx][:2]
        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            cat = self.target_transform(cat)
            
        return text, cat