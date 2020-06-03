
from data.textDataset import TextDataset
from utilities.mapping_cain import mapping_list
from time import time
from sklearn.model_selection import train_test_split

_script = "cain_mapping_cain_dp"
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

elif "ft".lower() in _ENCODER.lower():
    from encoders.FastTextEncoder import FTEncoder
    ENC = FTEncoder()

if "svm" in _CLASSIFIER.lower():
    from classifiers.SVM import CLASSIFIER
    CLF = CLASSIFIER()

def encoding(data):
    return ENC.process(data)

project_train = [
    {
        "name": "invoice",
        "label_path": r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\train\labels",
        "mapping_func": mapping_list["invoice"],
    },
    {
        "name": "bourbon",
        "label_path": r"D:\Workspace\cinnamon\data\Bourbon\train\labels",
        "mapping_func": mapping_list["bourbon"],
    },
    {
        "name": "sompo_holdings",
        "label_path": r"D:\Workspace\cinnamon\data\sompo_holdings\train\labels",
        "mapping_func": mapping_list["sompo_holdings"],
    },
    #{
    #    "name": "myl",
    #    "label_path": r"D:\Workspace\cinnamon\data\myl\labels",
    #    "mapping_func": mapping_list["myl"],
    #}
]

project_test = [
    {
        "name": "invoice",
        "label_path": r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\test\labels",
        "mapping_func": mapping_list["invoice"],
    },
    {
        "name": "bourbon",
        "label_path": r"D:\Workspace\cinnamon\data\Bourbon\test\labels",
        "mapping_func": mapping_list["bourbon"],
    },
    {
        "name": "sompo_holdings",
        "label_path": r"D:\Workspace\cinnamon\data\sompo_holdings\val\labels",
        "mapping_func": mapping_list["sompo_holdings"],
    },
]

trainset = TextDataset(labels_path=project_train, only=None)#only=["key", "value"])
testset = TextDataset(labels_path=project_test, only=None)#only=["key", "value"])
print(trainset.category)
print(len(trainset))

import random
trainset_filt = []
for item in trainset:
    if item[1] == '' and random.uniform(0, 1) > 0.02:
        continue
    
    trainset_filt.append(item)

print("LEN TRAIN FILT:", len(trainset_filt))


train_enc = encoding(trainset_filt)

if testset is None:
    # Split to data and label
    X = train_enc.drop('Class', axis=1)
    y = train_enc['Class']

    print(X.shape)
    # Split train - test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
else:
    X_train = train_enc.drop('Class', axis=1)
    y_train = train_enc['Class']
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.90)

    test_enc = encoding(testset)
    X_test = test_enc.drop('Class', axis=1)
    y_test = test_enc['Class']


# FITTING
print("FITTING===================")
CLF = CLASSIFIER()
CLF.fit(X_train, y_train)

# PREDICTION
s = time()
y_pred = CLF.predict(X_test)
print(f"Elapsed {time()-s} s")

# STORING MODEL
filename = _script + _ENCODER + _CLASSIFIER + '.sav'
CLF.save(filename)

# METRICS
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
