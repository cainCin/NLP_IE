import torch
import os

# =============== ENCODER
from encoders.Word2VecEncoder import W2VEncoder
ENC = W2VEncoder(weight_paths=r"D:\Workspace\cinnamon\code\github\NLP_IE\jawiki_20180420_100d.pkl")
encode_length = 100
_default = {
    "lr": 1e-4,
    "weight_path": "W2VCNN_1e4_demo_v0.pt",
}

## utility
nbf = encode_length
max_length = 128
nbl = max_length
for i in range(6):
    nbf = nbf // 2
    nbl = nbl // 2

max_epoch = 50

def transform(text, encode_engine=ENC, max_length=max_length):
    out = torch.zeros(size=(1, max_length, encode_length))
    enc = torch.Tensor(encode_engine.encode(text)).view(1, -1, encode_length)
    out[:, :enc.shape[1],:] = enc
    return out

category = ['', 'address', 'company_name', 'date', 'description', 'number', 'tel_fax', 'zipcode']
def target_transform(label, category=category):
    out = torch.zeros(size=(len(category), ))
    if len(label) == 0: #UNKNOWN
        out[0] = 1.
        return out

    for i, cat in enumerate(category):
        if i == 0: continue
        if cat in label:
            out[i] = 1.

    return out

# ============ Classifier
from classifiers.CNN import Net, train

from data.textDataset import TextDataset
from utilities.mapping import mapping_list
from sklearn.metrics import classification_report, confusion_matrix


# set up dataloader
project_train = [
    {
        "name": "invoice",
        "label_path": r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\train\labels",
        "mapping_func": mapping_list["invoice"],
    }
]
project_test = [
    {
        "name": "invoice",
        "label_path": r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\test\labels",
        "mapping_func": mapping_list["invoice"],
    }
]



net = Net(nClass=len(category))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

# load checkpoint
if os.path.isfile(_default.get('weight_path')):
    net.load_state_dict(torch.load(_default.get('weight_path')))
    net.eval()
    print(f"Load {_default.get('weight_path')}....................")
else:
    print("FROM SCRATCH =========================")

train(net, project_train, project_test, transform=transform, target_transform=target_transform, device=device, config=_default, only=["value"])

# load checkpoint
#net.load_state_dict(torch.load(_default.get('weight_path')))
#net.eval()
#print(f"Load {_default.get('weight_path')}...")
"""
testset = TextDataset(labels_path=project_test, only=["value"], 
                        transform=transform)
with torch.no_grad():
    y_test = []
    y_pred = []
    for (text, label) in testset:
        text = text.to(device)
        pred = net(text.unsqueeze(0)).cpu()
        # append to report
        y_pred.append(category[torch.argmax(pred)])
        y_test.append(label)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
"""