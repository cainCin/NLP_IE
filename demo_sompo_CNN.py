import sys
sys.path.append(r"D:\Workspace\cinnamon\code\dev\utilities")
from basic_utils import visualize, imread, load_json
import copy
import json
import os
import torch

max_length = 128
encode_length = 100
category = ['', 'amount', 'branch_name', 'company_name', 'date', 'key_amount', 'key_branch_name', 'key_date', 'key_mix', 'key_person_name', 'key_ratio', 'key_score', 'key_status', 'person_name', 'ratio', 'score', 'status']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(filename="classifiers/W2VCNN_test_1e4_sompo_v6.pt"):
    from classifiers.SVM import CLASSIFIER
    from classifiers.CNN_dev import Net
    net = Net(nClass=len(category))
    net = net.to(device)
    net.load_state_dict(torch.load(filename))
    net.eval()

    return net

def load_enc(weight_paths="jawiki_20180420_100d.pkl"):
    from encoders.Word2VecEncoder import W2VEncoder
    ENC = W2VEncoder(weight_paths=weight_paths)

    return ENC

def transform(text, encode_engine, max_length=max_length):
    out = torch.zeros(size=(1, max_length, encode_length))
    try:
        enc = torch.Tensor(encode_engine.encode(text)).view(1, -1, encode_length)
        out[:, :enc.shape[1],:] = enc
    except:
        return out
    return out

def predict(enc, classifier, data):
    y_pred = []
    with torch.no_grad():
        for item in data:
            data_enc = transform(item[0], enc)
            x = data_enc.to(device)
            y_logit = classifier(x.unsqueeze(0))
            y_pred.append(category[torch.argmax(y_logit.cpu())])
    return y_pred

def process_image(image_path, flax_path, enc, classifier, label_path=None, out_path="."):
    label = load_json(label_path, only=None) if label_path else None
    flax = json.load(open(flax_path, "r", encoding="utf-8"))

    for item in flax:
        print("ITEM:", item.get("text"), item.get("type"))
        if len(item.get("text")) == 0: continue
        y_pred = predict(enc, classifier, [(item.get("text"), None)])
        if "key" in y_pred[0]:
            item.update({
                "text": f"{y_pred[0]}:{item.get('text')}",
                "key_type": "key",
                "type": y_pred[0],
            })
        elif len(y_pred[0]) > 0:
            item.update({
                "text": f"{y_pred[0]}:{item.get('text')}",
                "key_type": "value",
                "type": y_pred[0],
            })
        print("====> PREDICT:", y_pred)



    
    # visualization
    image = imread(image_path)

    img = visualize(image, label=label, flax=flax)
    #img.show()
    img.save(out_path + "/" + os.path.basename(image_path) + ".png")

if __name__ == "__main__":
    
    test_image = r"【請求書】旅行者向けIoT実証実験（WatchPhone）_201709_0"
    image_dir = r"D:\Workspace\cinnamon\data\sompo_holdings\val\images"
    label_dir = r"D:\Workspace\cinnamon\data\sompo_holdings\val\labels"
    flax_dir = r"D:\Workspace\cinnamon\reports\prj\sompo\input_output\debugs"

    import glob
    image_list = glob.glob(image_dir + "/*")

    out_path = "sompo_debug"
    os.makedirs(out_path, exist_ok=True)
    # load model
    enc = load_enc()
    classifier = load_model()
    for image_path in image_list:
        image_name = ".".join(os.path.basename(image_path).split(".")[:-1])
        label_path = os.path.join(label_dir, image_name + ".json")
        flax_path = os.path.join(flax_dir, image_name, "ocr_output.json")
    
    
        process_image(image_path, flax_path, enc=enc, classifier=classifier, label_path=label_path, out_path=out_path)
        
    pass