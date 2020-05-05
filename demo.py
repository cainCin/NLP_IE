
from data.invoice_dataset import TextDataset
from time import time


_ENCODER = "w2v"
_CLASSIFIER = "SVM"

if "BoW".lower() in _ENCODER.lower():
    from encoders.BoWEncoder import BoWEncoder
    ENC = BoWEncoder(vocab=r"D:\Workspace\cinnamon\code\prj\invoice\data\invoice_phase3.5\corpus.json")
elif "bert" in _ENCODER.lower():
    from encoders.BERTEncoder import BERTEncoder
    ENC = BERTEncoder()
elif "w2v".lower() in _ENCODER.lower():
    from encoders.Word2VecEncoder import W2VEncoder
    ENC = W2VEncoder()

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


train_labels_path = r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\train\labels"
test_labels_path = r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\test\labels"
trainset = TextDataset(train_labels_path, category=None, target_transform=target_transform)
testset = TextDataset(test_labels_path, category=None, target_transform=target_transform)
print(trainset.category)
train_enc = encoding(trainset)

if testset is None:
    from sklearn.model_selection import train_test_split
    # Split to data and label
    X = train_enc.drop('Class', axis=1)
    y = train_enc['Class']

    print(X.shape)
    # Split train - test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
else:
    X_train = train_enc.drop('Class', axis=1)
    y_train = train_enc['Class']

    test_enc = encoding(testset)
    X_test = test_enc.drop('Class', axis=1)
    y_test = test_enc['Class']


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
