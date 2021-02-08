import json, csv


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def convert_json(obj):
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)
        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]
        elif hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
            return convert_json(obj.__name__)
        elif hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}
        return str(obj)


def txt_to_csv(fname):
    with open(fname + ".txt", 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split(",") for line in stripped if line)
        with open(fname + '.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('title', 'intro'))
            writer.writerows(lines)
