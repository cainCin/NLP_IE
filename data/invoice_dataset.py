from torch.utils.data import Dataset, DataLoader
import glob
from utilities.utils import load_json
import numpy as np
import torch

class TextDataset(Dataset):
    """Text dataset from json"""

    def __init__(self, labels_path, category=None, transform=None, target_transform=None, only=["key", "value"]):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.only = only
        self.data, self.category = self.load_data(labels_path, category)
        
        self.transform = transform
        self.target_transform = target_transform
        
    def load_data(self, labels_path, category=None):
        list_label = glob.glob(labels_path + "/*")
        data = []
        for label_path in list_label:
            #print(f"Processing {label_path}")
            CASIA_output = load_json(label_path)
            if self.only is not None:
                # fetch value only
                item = [(item.get("text"), item.get("type")) for item in CASIA_output \
                        if item.get("type") not in ["", None, "None"] and item.get('key_type') in self.only]
            else:
                # fetch full data
                item = []
                for item_ in CASIA_output:
                    if len(item_.get('text')) == 0: continue
                    if item_.get('key_type').lower() in ['key']:
                        item.append((item_.get('text'), "key_" + item_.get('type')))
                    elif item_.get('key_type').lower() in ['value']:
                        item.append((item_.get('text'), item_.get('type')))
                    else:
                        item.append((item_.get('text'), ''))
            #item = [(item.get("text"), item.get("type")) for item in CASIA_output if len(item.get("text")) > 0]
            if category:
                out = []
                for element in item:
                    for cat in category:
                        if cat in element[1]:
                            out.append(element)
                            continue
                item = out

            data.extend(item)
        
        category = list(np.unique([item[1] for item in data]))
        
        return data, category
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
      
        text, cat = self.data[idx]
        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            cat = self.target_transform(cat)
            
        return text, cat