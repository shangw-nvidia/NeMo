import json
import os

from sklearn.model_selection import train_test_split

FILE_RANGES = {
    "dstc8_single_domain": {"train": range(1, 44), "dev": range(1, 8), "test": range(1, 12)},
    "dstc8_multi_domain": {"train": range(44, 128), "dev": range(8, 21), "test": range(12, 35)},
    "dstc8_all": {"train": range(1, 128), "dev": range(1, 21), "test": range(1, 35)},
    "DEBUG": {"train": range(1, 2), "dev": range(1, 2), "test": range(1, 3)},
    "multiwoz": {"train": range(1, 2), "dev": range(1, 2), "test": range(1, 2)},
    "sgdplus_single_domain": {"train": range(1, 2), "dev": range(1, 2), "test": range(1, 2)},
    "sgdplus_all": {"train": range(1, 3), "dev": range(1, 3), "test": range(1, 3)},
}

dstc8_data_dir = "./sgd_orig"
dstc8_out_dir = "./sgd_plus"

os.makedirs(dstc8_out_dir, exist_ok=True)


def get_dialogue_file_name(task_name):
    train_file_range = FILE_RANGES[task_name]["train"]
    dev_file_range = FILE_RANGES[task_name]["dev"]
    test_file_range = FILE_RANGES[task_name]["test"]

    _file_ranges = {
        "train": train_file_range,
        "dev": dev_file_range,
        "test": test_file_range,
    }

    dialog_paths = []
    for dataset in ["train", "dev", "test"]:
        dialog_paths.extend(
            [os.path.join(dstc8_data_dir, dataset, "dialogues_{:03d}.json".format(i)) for i in _file_ranges[dataset]]
        )
    return dialog_paths


def load_dialouges(dialog_paths):
    dialogs = []
    for dialog_json_filepath in sorted(dialog_paths):
        with open(dialog_json_filepath, 'r') as f:
            dialogs.extend(json.load(f))
    return dialogs


def load_schemas(dstc8_data_dir):
    schema_paths = []
    for dataset in ["train", "dev", "test"]:
        schema_paths.append(os.path.join(dstc8_data_dir, dataset, "schema.json"))

    schemas = []
    schemas_cleaned = []
    schemas_list = {}
    for schmea_json_filepath in sorted(schema_paths):
        with open(schmea_json_filepath, 'r') as f:
            schemas.extend(json.load(f))

    for schema in schemas:
        if schema["service_name"] not in schemas_list:
            schemas_list[schema["service_name"]] = True
            schemas_cleaned.append(schema)
    return schemas_cleaned


dialog_paths_single = get_dialogue_file_name(task_name="dstc8_single_domain")
dialog_paths_multi = get_dialogue_file_name(task_name="dstc8_multi_domain")

dialogs_single = load_dialouges(dialog_paths_single)
dialogs_multi = load_dialouges(dialog_paths_multi)
schemas = load_schemas(dstc8_data_dir)


for dataset in ["train", "dev", "test"]:
    folder_path = os.path.join(dstc8_out_dir, dataset)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(dstc8_out_dir, dataset, "schema.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(schemas, f, ensure_ascii=False, indent=2)


X = {}
X["train"], X["dev"] = train_test_split(dialogs_single, test_size=0.15, random_state=2020)
X["train"], X["test"] = train_test_split(X["train"], test_size=15.0 / 85.0, random_state=2020)

last_id = {}
last_id["train"] = 1  # FILE_RANGES["dstc8_all"]["train"][-1] + 1
last_id["dev"] = 1  # FILE_RANGES["dstc8_all"]["dev"][-1] + 1
last_id["test"] = 1  # FILE_RANGES["dstc8_all"]["test"][-1] + 1

for dataset in ["train", "dev", "test"]:
    for dial_id, dial in enumerate(X[dataset]):
        if len(str(dial_id)) > 5:
            print("Too many dialogues!")
            exit(1)
        idf = format(dial_id, "05")
        dial["dialogue_id"] = f"{last_id[dataset]}_{idf}"
    file_path = os.path.join(dstc8_out_dir, dataset, "dialogues_{:03d}.json".format(last_id[dataset]))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(X[dataset], f, ensure_ascii=False, indent=2)

X = {}
X["train"], X["dev"] = train_test_split(dialogs_multi, test_size=0.15, random_state=2020)
X["train"], X["test"] = train_test_split(X["train"], test_size=15.0 / 85.0, random_state=2020)

last_id = {}
last_id["train"] = 2  # FILE_RANGES["dstc8_all"]["train"][-1] + 2
last_id["dev"] = 2  # FILE_RANGES["dstc8_all"]["dev"][-1] + 2
last_id["test"] = 2  # FILE_RANGES["dstc8_all"]["test"][-1] + 2

for dataset in ["train", "dev", "test"]:
    for dial_id, dial in enumerate(X[dataset]):
        if len(str(dial_id)) > 5:
            print("Too many dialogues!")
            exit(1)
        idf = format(dial_id, "05")
        dial["dialogue_id"] = f"{last_id[dataset]}_{idf}"
    file_path = os.path.join(dstc8_out_dir, dataset, "dialogues_{:03d}.json".format(last_id[dataset]))
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(X[dataset], f, ensure_ascii=False, indent=2)
