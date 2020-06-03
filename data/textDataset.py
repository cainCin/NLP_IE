from .baseDataset import BaseDataset
import numpy as np

class TextDataset(BaseDataset):
    """
    change labels_path to list of dictionary containing:
    - project name
    - project mapping function
    - project label_path
    """
    def __init__(self, *args, **kwargs):
        super(TextDataset, self).__init__(*args, **kwargs)

    def load_data(self, projects_path, key_list=None):
        if isinstance(projects_path, list):
            data = []
            for project_path in projects_path:
                print(f"Fetching data for {project_path.get('name')}")
                # fetch data
                fetched_data = self.fetch_data(project_path.get('label_path'), category=key_list)

                # mapping
                # mapped_data = [(_text, project_path.get("mapping_func")[_type], _key_type) for _text, _type, _key_type, label_path in fetched_data]
                mapped_data = []
                for (_text, _type, _key_type, label_path) in fetched_data:
                    _text = str(_text)
                    try:
                        new_type = project_path.get("mapping_func")[_type]
                        if _key_type in ["key"]:
                            if new_type in ['address', 'company_name', 'zipcode']:
                                new_type = ''
                            elif len(new_type) > 0:
                                new_type = _key_type + "_" + new_type

                    except:
                        mapped_data.append((_text, '', _key_type, label_path))
                        print(f"Error in mapping at {label_path}: formal_key {_type} not in the map list")
                    else:
                        mapped_data.append((_text, new_type, _key_type, label_path))

                

                # append to output
                data.extend(mapped_data)
        else:
            print("labels_path should contain dict with project name, project mapping function, project label_path")
            return [], []

        category = list(np.unique([item[1] for item in data]))
        
        return data, category