import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.optim as optim

import sys
sys.path.append("..")

from data.textDataset import TextDataset
# setup encoder
from encoders.Word2VecEncoder import W2VEncoder
ENC = W2VEncoder(weight_paths=r"D:\Workspace\cinnamon\code\github\NLP_IE\jawiki_20180420_100d.pkl")
encode_length = 100
_default = {
    "lr": 1e-4,
    "weight_path": "W2VCNN_test_1e4.pt",
}

nbf = encode_length
max_length = 128
nbl = max_length
for i in range(6):
    nbf = nbf // 2
    nbl = nbl // 2

max_epoch = 300


#from encoders.BERTEncoder import BERTEncoder
#ENC = BERTEncoder()
#encode_length = 768
#_default = {
#    "lr": 1e-4,
#    "weight_path": "BERTCNN_test_1e4.pt",
#}



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

class Net(nn.Module):
    def __init__(self):
        pass

    def __init__(self, nClass=2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)     # 64, 100
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)    # 32, 50
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)    # 16, 25
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)    # 8, 12
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)    # 4, 6
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)    # 2, 3
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * nbf * nbl, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, nClass)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        #x = torch.mean(x, dim=2)
        x = x.view(-1, 256 * nbl * nbf)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

def val(model, dataloader, is_train=False, device=torch.device("cpu")):
    acc = 0
    nbs = 0
    nbt = 20 if is_train else len(dataloader)
    for data, label in dataloader:
    #with tqdm.trange(nbt) as t:
    #    data_iter = iter(dataloader)
    #    for iteration in t:
    #        data, label =  next(data_iter)
        data = data.to(device)
        label = label.to(device)
        # classification
        out = model(data)
        
        # get acc
        pred = torch.argmax(out.detach(), dim=1)
        truth = torch.argmax(label, dim=1)

        acc += sum(pred==truth)
        nbs += len(pred)

    return (acc * 1. / nbs)

def train(net, project_train, project_test, transform=None, target_transform=None, device=torch.device("cpu"), config=_default, only=["value"]):
    # displaying configuration
    print("TRAINING CONFIG:")
    for key, value in config.items(): print(f"{key}: {value}")

    trainset = TextDataset(labels_path=project_train, only=only, 
                            transform=transform, 
                            target_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=1)

    testset = TextDataset(labels_path=project_test, only=only, 
                            transform=transform, 
                            target_transform=target_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=1)


    #net = Net(nClass=len(category))
    #net = net.cuda()
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=config.get('lr'))# SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_acc = val(net, trainloader, is_train=True, device=device)
    val_acc = val(net, testloader, is_train=False, device=device)
    print("VAL===================")
    best_val_acc = val_acc
    for epoch in range(max_epoch):
        #data_iter = iter(trainloader)
        #with tqdm.trange(len(trainloader)) as t:
        #    for iteration in t:
        #        texts, labels = next(data_iter)
        print("EPOCH", epoch)
        for i, (texts, labels) in enumerate(trainloader):
            # use cuda
            data = texts.to(device)
            label = labels.to(device)
            # classification
            out = net(data)
            #print(f"IN {data.shape}, Out {out.shape}, LABEL {label.shape}")
            # loss and update
            loss = criterion(out, label)
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if i % 5:
            #print(f"[{epoch}][{i}/{len(trainloader)}] == BCE Loss: {loss.item()}, TRAIN ACC: {train_acc}, VAL ACC: {val_acc}")

            
        

        # save weight
        #if (iteration > 0) and (iteration % 1000 == 0): torch.save(net.state_dict(), config.get('weight_path'))

            #t.set_description(f"BCE Loss: {loss.item()}, TRAIN ACC: {train_acc}, VAL ACC: {val_acc}")
        
        # evaluation
        train_acc = val(net, trainloader, is_train=True, device=device)
        val_acc = val(net, testloader, is_train=False, device=device)
        print(f"[{epoch}] == BCE Loss: {loss.item()}, TRAIN ACC: {train_acc}, VAL ACC: {val_acc}")
        # save weight
        if val_acc > best_val_acc:
            torch.save(net.state_dict(), config.get('weight_path'))
            best_val_acc = val_acc
            print("WEIGHT STORING ...")

if __name__ == "__main__":
    from utilities.mapping import mapping_list
    from sklearn.metrics import classification_report, confusion_matrix
    import os

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
        print(f"Load {_default.get('weight_path')}...")
    else:
        print("FROM SCRATCH=====================")

    #train(net, project_train, project_test, device=device)
    train(net, project_train, project_test, transform=transform, target_transform=target_transform, device=device, config=_default, only=["value"])
    # load checkpoint
    #net.load_state_dict(torch.load(_default.get('weight_path')))
    #net.eval()
    #print(f"Load {_default.get('weight_path')}...")
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

        
        


    
    


    pass