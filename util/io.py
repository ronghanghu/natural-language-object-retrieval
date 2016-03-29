import json

def load_str_list(filename):
    with open(filename, 'r') as f:
        str_list = f.readlines()
    str_list = [s[:-1] for s in str_list]
    return str_list

def save_str_list(str_list, filename):
    str_list = [s+'\n' for s in str_list]
    with open(filename, 'w') as f:
        f.writelines(str_list)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(json_obj, filename):
    with open(filename, 'w') as f:
        json.dump(json_obj, f, separators=(',\n', ':\n'))
