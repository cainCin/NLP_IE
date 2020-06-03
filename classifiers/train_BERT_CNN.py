from data import TextDataset
from models.BERTEncoder import TextEncoder
import torch.nn as nn
import torch
import torch.nn.functional as F
import tqdm
import torch.optim as optim
import time

nb_iter = 15000
category = ["name", "date", "quantity", "number", "type", "zip", "amount", "unit", "tel"]
train_path = "/mnt/sda1/backup/lapCin/Workspace/Cinnamon/Data/Invoice/Phase 3/train/labels"
test_path = "/mnt/sda1/backup/lapCin/Workspace/Cinnamon/Data/Invoice/Phase 3/test/labels"

_default = {
    "lr": 1e-4,
    "weight_path": "BERTCNN_test_1e4.pt",
}

encode_engine = TextEncoder()

def transform(text, max_size=64):
    out = torch.zeros(size=(1, max_size, 768))
    enc = encode_engine.encode(text)
    out[:,:enc.shape[1],:] = enc
    return out


def target_transform(label, category=category):
    out = torch.zeros(size=(len(category), ))
    if label is None: return out
    for i, cat in enumerate(category):
        if cat in label:
            out[i] = 1.

    return out

def val(model, dataloader, is_train=False, device=torch.device("cpu")):
    acc = 0
    nbs = 0
    nbt = 20 if is_train else len(dataloader)
    with tqdm.trange(nbt) as t:
        data_iter = iter(dataloader)
        for iteration in t:
            raw_texts, data, label =  next(data_iter)
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


class Net(nn.Module):
    def __init__(self):
        pass

    def __init__(self, nClass=2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)     # 64, 768
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)    # 32, 384
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)    # 16, 192
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)    # 8, 96
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)    # 4, 48
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)    # 2, 24
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 12, 120)
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
        x = x.view(-1, 256 * 1 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

def predict(model, text_list, mode="text"):
    if mode == "text":
        dataset = TextDataset(labels_path=None, category=category, transform=transform, target_transform=target_transform)
        dataset.data = [(text, None) for text in text_list]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                            shuffle=False, num_workers=1)
        for text, data, label in dataloader:
            data = data.cuda()
            out = model(data)
            pred = torch.argmax(out.detach(), dim=1).cpu()

            for i in range(len(text)):
                print(text[i], category[pred[i]])
    else:
        x_text = []
        y_pred = []
        y_true = []
        
        for label_path in text_list:
            dataset = TextDataset(labels_path=label_path, category=category, transform=transform, target_transform=target_transform)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                shuffle=False, num_workers=1)
            data_iter = iter(dataloader)
            with tqdm.trange(len(dataloader)) as t:
                for iteration in t:
                #for text, data, label in dataloader:
                    text, data, label = next(data_iter)
                    data = data.cuda()
                    out = model(data)
                    pred = torch.argmax(out.detach(), dim=1).cpu()

                    """
                    for i in range(len(text)):
                        print(text[i], category[pred[i]])
                    """
                    for i in range(len(text)):
                        pred_label = category[pred[i]]
                        true_label = [c for k, c in enumerate(category) if label[i,k] > 0.]
                        if len(true_label) == 0:
                            true_label = ""
                        elif pred_label in true_label:
                            true_label = pred_label
                        else:
                            true_label = true_label[0]

                        x_text.append(text[i])
                        y_pred.append(pred_label)
                        y_true.append(true_label)
        
        return x_text, y_pred, y_true
    

def train(net, train_path, test_path, device=torch.device("cpu"), config=_default):
    # displaying configuration
    print("TRAINING CONFIG:")
    for key, value in config.items(): print(f"{key}: {value}")

    trainset = TextDataset(labels_path=train_path, category=category, transform=transform, target_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=1)

    testset = TextDataset(labels_path=test_path, category=category, transform=transform, target_transform=target_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=1)


    #net = Net(nClass=len(category))
    #net = net.cuda()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=config.get('lr'))# SGD(net.parameters(), lr=0.001, momentum=0.9)


    data_iter = iter(trainloader)
    train_acc = val(net, trainloader, is_train=True, device=device)
    val_acc = val(net, testloader, is_train=False, device=device)
    with tqdm.trange(nb_iter) as t:
        for iteration in t:
            try:
                raw_texts, texts, labels = next(data_iter)
            except:
                train_acc = val(net, trainloader, is_train=True, device=device)
                val_acc = val(net, testloader, is_train=False, device=device)
                data_iter = iter(trainloader)
                raw_texts, texts, labels = next(data_iter)

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

            # save weight
            if (iteration > 0) and (iteration % 1000 == 0): torch.save(net.state_dict(), config.get('weight_path'))

            t.set_description(f"BCE Loss: {loss.item()}, TRAIN ACC: {train_acc}, VAL ACC: {val_acc}")
        
if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net = Net(nClass=len(category))
    net = net.to(device)
    config = {
        "lr": 1e-4,
        "weight_path": "BERTCNN_test_1e4.pt",
    }
    weight_path = "BERTCNN_test.pt"

    ### TRAIN
    if False:
        # load pretrain weight
        net.load_state_dict(torch.load(weight_path))
        net.eval()
        print(f"Loading pretrain {weight_path}")
        
        # train
        train(net, train_path, test_path, device=device, config=config)
    ### VAL
    if True:
        # load weight
        net.load_state_dict(torch.load(config.get('weight_path')))
        net.eval()
        print("Load BERTCNN_test.pt...")

        # load test set
        testset = TextDataset(labels_path=test_path, category=category, transform=transform, target_transform=target_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                                    shuffle=False, num_workers=1)

        # eval
        print(f"Starting EVALUATION on {test_path}")
        start_time = time.time()
        val_acc = val(net, testloader, is_train=False, device=device)
        end_time = time.time()

        print(f"Elapsed {end_time-start_time}s for {val_acc} acc... ==> {(end_time-start_time)/len(testset)} [s/sentence]")

    ### INFERENCE
    if False:
        # load weight
        net.load_state_dict(torch.load("BERTCNN_test.pt"))
        net.eval()
        print("Load BERTCNN_test.pt...")

        predict(net, ["1990/03/25", "200", "200.00"])

