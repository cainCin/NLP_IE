import sys
sys.path.append(r"D:\Workspace\cinnamon\code\dev\utilities")
from basic_utils import visualize, imread, load_json
import copy
import json
import os
import torch

max_length = 128
encode_length = 100
category = ['', 'address', 'company_name', 'date', 'description', 'key_date', 'key_description', 'key_number', 'key_tel_fax', 'number', 'tel_fax', 'zipcode']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(filename="classifiers/W2VCNN_test_1e4_draft.pt"):
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
    enc = torch.Tensor(encode_engine.encode(text)).view(1, -1, encode_length)
    out[:, :enc.shape[1],:] = enc
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

def process_image(image_path, flax_path, enc, classifier, label_path=None):
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
    img.show()
    img.save(os.path.basename(image_path) + ".png")

if __name__ == "__main__":
    
    test_image = r"【請求書】旅行者向けIoT実証実験（WatchPhone）_201709_0"
    image_path = r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\test\images\{}.jpg".format(test_image)
    label_path = r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\test\labels\{}.json".format(test_image)
    flax_path = r"D:\Workspace\cinnamon\code\prj\invoice\200227\debug_outputs\{}\1\kv_input.json".format(test_image)
    
    
    # daiichi
    #image_path = r"D:\Workspace\cinnamon\data\daiichi\full_data\0_004992010109557022341000160001002-20181203120957000-00000000-009-C200O.png"
    #label_path = r"D:\Workspace\cinnamon\data\daiichi\full_data\0_004992010109557022341000160001002-20181203120957000-00000000-009-C200O.json"
    #flax_path = r"D:\Workspace\cinnamon\code\dev\daiichi\0_004992010109557022341000160001002-20181203120957000-00000000-009-C200O.la_ocr.json"
    
    
    # sumitoma
    #image_path = r"D:\Workspace\cinnamon\data\sumimota\images\2018030500875_558404_04_20180305l0015000550_1.png"
    #label_path = r"D:\Workspace\cinnamon\data\sumimota\labels\2018030500875_558404_04_20180305l0015000550_1.json"
    #flax_path = r"D:\Workspace\cinnamon\code\dev\sumimota\2018030500875_558404_04_20180305l0015000550_1.la_ocr.json"

    # MYL
    #image_path = r"D:\Workspace\cinnamon\data\myl\images\img00060003.jpg.0.Blacken.jpg"
    #label_path = r"D:\Workspace\cinnamon\data\myl\labels\img00060003.jpg.0.Blacken.json"
    #flax_path = r"D:\Workspace\cinnamon\code\dev\test\img00060003.jpg.0.Blacken.la_ocr.json"
    
    #image_path = r"D:\Workspace\cinnamon\code\dev\samples\Understand-Water-Sewage-Bill-Japan-Kanji-Cheat-Sheet-Real-Estate-Japan-1024x614.jpg"
    #label_path = None
    #flax_path = r"D:\Workspace\cinnamon\code\dev\samples\output\Understand-Water-Sewage-Bill-Japan-Kanji-Cheat-Sheet-Real-Estate-Japan-1024x614.la_ocr.json"

    # load model
    enc = load_enc()
    classifier = load_model()
    process_image(image_path, flax_path, enc=enc, classifier=classifier, label_path=label_path)
    pass