
from data.invoice_dataset import TextDataset
from time import time
from sklearn.model_selection import train_test_split

script = "EVAL_"
_ENCODER = "bow"
_CLASSIFIER = "SVM"
REDUCE_UNKNOWN = False

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


category = ["name", "date", "number", "amount", "address"]
def target_transform(label, category=category):
    for cat in category:
        if cat in label:
            return cat
    if "tax" in label:
        return "amount"
    #if "fax" in label:
    #    return "tel"
    if "quantity" in label:
        return "number"
    return ""


train_labels_path = r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\train\labels"
test_labels_path = r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\test\labels"
trainset = TextDataset(train_labels_path, category=None, target_transform=target_transform, only=["value"])
testset = TextDataset(test_labels_path, category=None, target_transform=target_transform, only=["value"])
print(trainset.category)
print(len(trainset))

print("ENCODING===================")
# Reduce the number of unknown data
if REDUCE_UNKNOWN:
    import random
    trainset_filt = []
    for item in trainset:
        if item[1] == '' and random.uniform(0, 1) > 0.2:
            continue
        
        trainset_filt.append(item)

    print("LEN TRAIN FILT:", len(trainset_filt))
else:
    trainset_filt = trainset

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
filename = script + _ENCODER + _CLASSIFIER + '.sav'
CLF.save(filename)

# METRICS
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
