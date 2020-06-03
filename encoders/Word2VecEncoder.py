#from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd
import numpy as np

from wikipedia2vec import Wikipedia2Vec

class W2VEncoder:
    def __init__(self, weight_paths='jawiki_20180420_100d.pkl'):
        self.wiki2vec = Wikipedia2Vec.load(weight_paths)
        pass

    def compress(self, X, mode="average"):
        return np.mean(X, axis=0)

    def encode(self, text):
        X = []
        for c in text:
            try:
                X.append(self.wiki2vec.get_word_vector(c.lower()))
            except:
                continue
        if len(X) > 0:
            return X
            # X = list(self.compress(X))
        else:
            X = None
        return X

    def export_pd(self, X):
        return pd.DataFrame(X)

    def process(self, data):
        text, category = [], []
        for (t, c) in data:
            text.append(t)
            category.append(c)


        X = [self.encode(t) for t in text]
        X = [list(self.compress(x)) for x in X]
        category = [c for item, c in zip(X, category) if item is not None]
        X = [item for item in X if item is not None]
        X = self.export_pd(X)
        X["Class"] = category
        return X


if __name__ == "__main__":
    text = ["日本語（にほんご、にっぽんご[注 1]）は、主に日本国内や日本人同士の間で使用されている言語である。"]
    enc = W2VEncoder()
    #print(enc.encode(text))
    X = enc.process([("abc", None), ("b", None)])
    print(X)