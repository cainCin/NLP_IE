from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd

class BoWEncoder:
    def __init__(self, vocab=None):
        if vocab is None:
            self.vectorizer = CountVectorizer(analyzer="char")
        else:
            if isinstance(vocab, str):
                if vocab.split(".")[-1] == "json":
                    vocab = json.load(open(vocab, "r", encoding="utf-8"))
                    self.vocab = [c for c in vocab]
                    # print(vocab)
                else:
                    vocab = open(vocab, "r", encoding="utf-8").readlines()
                    self.vocab = vocab.split("\n")
            self.vectorizer = CountVectorizer(analyzer="char", vocabulary=self.vocab)
        pass

    def encode(self, text):
        if isinstance(text, str):
            X = self.vectorizer.fit_transform(text).toarray()
        elif isinstance(text, list):
            X = self.vectorizer.fit_transform(text).toarray()

        return X

    def export_pd(self, X):
        return pd.DataFrame(X)

    def process(self, data):
        text, category = [], []
        for (t, c) in data:
            text.append(t)
            category.append(c)

        X = self.encode(text)
        X = self.export_pd(X)
        X["Class"] = category
        return X


if __name__ == "__main__":
    text = ["abc", "xyztuv"]
    enc = BoWEncoder(vocab=r"D:\Workspace\cinnamon\code\prj\invoice\data\invoice_phase3.5\corpus.json")
    #print(enc.encode(text))
    X = enc.process([("a", None), ("b", None)])
    print(X)