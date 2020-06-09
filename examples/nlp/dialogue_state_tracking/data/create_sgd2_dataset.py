import json
import os

from sklearn.model_selection import train_test_split

FILE_RANGES = {
    "dstc8_single_domain": {"train": range(1, 44), "dev": range(1, 8), "test": range(1, 12)},
    "dstc8_multi_domain": {"train": range(44, 128), "dev": range(8, 21), "test": range(12, 35)},
    "dstc8_all": {"train": range(1, 128), "dev": range(1, 21), "test": range(1, 35)},
    "DEBUG": {"train": range(1, 2), "dev": range(1, 2), "test": range(1, 3)},
    "multiwoz": {"train": range(1, 2), "dev": range(1, 2), "test": range(1, 2)},
    "sgdplus_single_domain": {"train": range(128, 129), "dev": range(21, 22), "test": range(35, 36)},
    "sgdplus_all": {"train": range(128, 130), "dev": range(21, 23), "test": range(35, 37)},
}

dstc8_data_dir = "./sgd"


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


dialog_paths_single = get_dialogue_file_name(task_name="dstc8_single_domain")
dialog_paths_multi = get_dialogue_file_name(task_name="dstc8_multi_domain")

dialogs_single = load_dialouges(dialog_paths_single)
dialogs_multi = load_dialouges(dialog_paths_multi)

X_train, X_dev = train_test_split(dialogs_single, test_size=0.15, random_state=2020)
X_train, X_test = train_test_split(X_train, test_size=15.0 / 85.0, random_state=2020)

last_train_id = FILE_RANGES["dstc8_all"]["train"][-1] + 1
last_dev_id = FILE_RANGES["dstc8_all"]["dev"][-1] + 1
last_test_id = FILE_RANGES["dstc8_all"]["test"][-1] + 1


file_path = os.path.join(dstc8_data_dir, "train", "dialogues_{:03d}.json".format(last_train_id))
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(X_train, f, ensure_ascii=False, indent=2)

file_path = os.path.join(dstc8_data_dir, "dev", "dialogues_{:03d}.json".format(last_dev_id))
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(X_dev, f, ensure_ascii=False, indent=2)

file_path = os.path.join(dstc8_data_dir, "test", "dialogues_{:03d}.json".format(last_test_id))
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(X_test, f, ensure_ascii=False, indent=2)


X_train, X_dev = train_test_split(dialogs_multi, test_size=0.15, random_state=2020)
X_train, X_test = train_test_split(X_train, test_size=15.0 / 85.0, random_state=2020)

last_train_id = FILE_RANGES["dstc8_all"]["train"][-1] + 2
last_dev_id = FILE_RANGES["dstc8_all"]["dev"][-1] + 2
last_test_id = FILE_RANGES["dstc8_all"]["test"][-1] + 2


file_path = os.path.join(dstc8_data_dir, "train", "dialogues_{:03d}.json".format(last_train_id))
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(X_train, f, ensure_ascii=False, indent=2)

file_path = os.path.join(dstc8_data_dir, "dev", "dialogues_{:03d}.json".format(last_dev_id))
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(X_dev, f, ensure_ascii=False, indent=2)

file_path = os.path.join(dstc8_data_dir, "test", "dialogues_{:03d}.json".format(last_test_id))
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(X_test, f, ensure_ascii=False, indent=2)
