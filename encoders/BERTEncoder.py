import torch
from transformers import *
import pandas as pd

class BERTEncoder:
    _default = {
            "MODELS": [(BertModel,       BertTokenizer,       'bert-base-uncased')
         ],
    }
    def __init__(self):
        self._set_default()
        pass
    
    def _set_default(self):
        for model_class, tokenizer_class, pretrained_weights in self._default["MODELS"]:
            # Load pretrained model/tokenizer
            self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
            self.model = model_class.from_pretrained(pretrained_weights)
    
    def encode(self, text, special_tokens=True):
        input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=special_tokens)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
    
    def compress(self, X, mode="average"):
        return X.mean(dim=1)
    
    def export_pd(self, X):
        return pd.DataFrame(X)
        
    def process_text_list(self, text_list):
        X = [self.encode(text) for text in text_list]
        X = [self.compress(x) for x in X]
        X = torch.cat(X, dim=0).numpy()
        X = self.export_pd(X)
        return X
        
    def process(self, data):
        text_list = [text for (text, C) in data]
        X = self.process_text_list(text_list)
        category = [C for text, C in data]
        
        X["Class"] = category
        return X

if __name__ == "__main__":
    text = ["abc", "xyztuv"]
    enc = BERTEncoder()
    print(enc.encode(text).shape)
