
from data.invoice_dataset import TextDataset
from time import time


_ENCODER = "bert"
_CLASSIFIER = "SVM"

if "BoW".lower() in _ENCODER.lower():
    from encoders.BoWEncoder import BoWEncoder
    ENC = BoWEncoder(vocab=r"D:\Workspace\cinnamon\code\prj\invoice\data\invoice_phase3.5\corpus.json")
elif "bert" in _ENCODER.lower():
    from encoders.BERTEncoder import BERTEncoder
    ENC = BERTEncoder()

if "svm" in _CLASSIFIER.lower():
    from classifiers.SVM import CLASSIFIER
    CLF = CLASSIFIER()

def encoding(data):
    return ENC.process(data)


category = ["name", "date", "type", "quantity", "amount"]
def target_transform(label, category=category):
    for cat in category:
        if cat in label:
            return cat
    return ""


labels_path = r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\train\labels"
trainset = TextDataset(labels_path, category, target_transform=target_transform)
print(trainset.category)
out = encoding(trainset)

from sklearn.model_selection import train_test_split
# Split to data and label
X = out.drop('Class', axis=1)
y = out['Class']

print(X.shape)
# Split train - test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# FITTING
CLF = CLASSIFIER()
CLF.fit(X_train, y_train)
s = time()
y_pred = CLF.predict(X_test)
print(f"Elapsed {time()-s} s")

# METRICS
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
