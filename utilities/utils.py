
import json

def load_json(label_path):
    CASIA_output = []
    with open(label_path, "r", encoding="utf-8") as f:
        label = json.load(f)
        for info in label['attributes']['_via_img_metadata']['regions']:
            shape_info = info['shape_attributes']
            item_info = info['region_attributes']
            text = item_info.get('label')
            formal_key = item_info.get('formal_key')
            key_type = item_info.get('key_type')
            try:
                x,y,w,h = [shape_info['x'], shape_info['y'], shape_info['width'], shape_info['height']]
                loc = [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]
            except:
                loc = [(x,y) for (x,y) in zip(shape_info['all_points_x'], shape_info['all_points_y'])]
            #print(f"{text}: - KEY: {formal_key} - TYPE: {key_type} - LOC: {x,y,w,h}")
            
            # put into CASIA
            item = {
                'text': text,
                'type': formal_key,
                'key_type': key_type,
                'location': loc,
            }
            CASIA_output.append(item)
    return CASIA_output

