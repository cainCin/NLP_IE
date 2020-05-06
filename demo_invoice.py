import sys
sys.path.append(r"D:\Workspace\cinnamon\code\dev\utilities")
from basic_utils import visualize, imread, load_json
import copy
import json

def load_model(filename="w2vSVM.sav"):
    from classifiers.SVM import CLASSIFIER
    CLF = CLASSIFIER()
    CLF.load(filename)
    return CLF

def load_enc():
    from encoders.Word2VecEncoder import W2VEncoder
    ENC = W2VEncoder()

    return ENC

def fetch_data(label):
    data = []
    for item in label:
        data.append((item.get('text'), item.get('type')))
    return data

def predict(enc, classifier, data):
    data_enc = enc.process(data)
    X = data_enc.drop('Class', axis=1)
    y = data_enc['Class']

    y_pred = classifier.predict(X)
    return y_pred
    

if __name__ == "__main__":
    image_path = r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\test\images\【TIS様】Pitch Tokyo請求書 (Aniwo)_0.jpg"
    label_path = r"D:\Workspace\cinnamon\data\invoice\Phase 3.5\test\labels\【TIS様】Pitch Tokyo請求書 (Aniwo)_0.json"

    label = load_json(label_path, only=None)
    
    #flax = copy.deepcopy(label)
    flax_path = r"D:\Workspace\cinnamon\code\prj\invoice\200227\debug_outputs\【TIS様】Pitch Tokyo請求書 (Aniwo)_0\1\kv_input.json"
    flax = json.load(open(flax_path, "r", encoding="utf-8"))
    # load model
    enc = load_enc()
    classifier = load_model()

    # predict
    for item in flax:
        print("ITEM:", item.get("text"), item.get("type"))
        if len(item.get("text")) == 0: continue
        y_pred = predict(enc, classifier, [(item.get("text"), None)])
        item.update({
            "text": f"{y_pred[0]}: {item.get('text')}",
            "key_type": "value",
        })
        print("====> PREDICT:", y_pred)



    
    # visualization
    image = imread(image_path)

    img = visualize(image, label=label, flax=flax)
    img.show()
    img.save("demo.png")
    pass