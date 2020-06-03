
import json

def get_boundingbox(box_shape_info):
    if box_shape_info["name"] == "rect":
        x, y, w, h = box_shape_info['x'], box_shape_info['y'], box_shape_info['width'], box_shape_info['height']
        cnt = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    else:
        cnt = [(x, y) for (x, y) in zip(box_shape_info["all_points_x"], box_shape_info["all_points_y"])]
    
    return cnt

def load_json(json_path, only=["key", "value"], key_list=None, addition_attr=[]):
    out = []
    with open(json_path, "r", encoding="utf-8") as f:
        label_info = json.load(f)

        for info in label_info['attributes']['_via_img_metadata']['regions']:
            ## QA info
            box_shape_info = info['shape_attributes']
            box_attributes_info = info['region_attributes']

            # check condition of key
            key = box_attributes_info.get("formal_key")
            
            if key is None: continue # key is missing
            if key_list is not None:
                get_info = key in key_list
            else:
                get_info = True
            
            ## predict if labelled
            if get_info:
                try:
                    ## Label info
                    key = box_attributes_info['formal_key']
                    target = box_attributes_info['label']

                    key_type = box_attributes_info.get("key_type") if box_attributes_info.get("key_type") is not None \
                                    else box_attributes_info.get("type")

                    if only:
                        if key_type not in only: continue
                    
                    ## Layout info
                    cnt = get_boundingbox(box_shape_info)#box_shape_info['x'], box_shape_info['y'], box_shape_info['width'], box_shape_info['height']

                except:
                    print(f"Error in {info}")

            # add to ouput
            out_item = {
                "type": key,
                "text": target,
                "key_type": key_type,
                "location": cnt,
            }

            # add additional attribution
            for attr in addition_attr:
                out_item.update({
                    attr: box_attributes_info.get(attr)
                })

            out.append(out_item)

    return out

