from pathlib import Path


def get_config():
    from pathlib import Path
    return {
        "batch_size": 8,
        "num_epochs": 10,
        "lr": 1e-4,
        "seq": 350,
        "d_model": 512,
        "train_csv": "dataset_train.csv",
        "lang_src": "hindi",
        "lang_tgt": "malayalam",
        "preload": None,
        "model_basename": "tmodel_",
        "artifact_dir": "artifacts",
        "experiment_name": "runs/tmodel",
        "vocab_size": 8000,
    }


def get_weights_file_path(config, epoch: str):
    model_folder = Path(config["model_folder"])
    model_folder.mkdir(parents=True, exist_ok=True)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(model_folder / model_filename)


def latest_weights_file_path(config):
    model_folder = Path(config["model_folder"])
    if not model_folder.exists():
        return None

    weights_files = list(model_folder.glob(f"{config['model_basename']}*"))
    if len(weights_files) == 0:
        return None

    weights_files.sort()
    return str(weights_files[-1])
# ###
# from pathlib import Path

# def get_config():
#     return {
#         "batch_size": 8,
#         "num_epochs": 10,
#         "lr": 10**-4,
#         "seq": 350,
#         "d_model": 512,
#         "datasource": 'YADHU1234/merged_hindi_malayalam_dataset_hugging_face_without_in22.csv',
#         "lang_tgt": "malayalam",
#         "lang_src": "hindi",
#         "model_folder": "weights",
#         "model_basename": "tmodel_",
#         "preload": "latest",
#         "tokenizer_file": "tokenizer_{0}.json",
#         "experiment_name": "runs/tmodel"
#     }

# def get_weights_file_path(config, epoch: str):
#     model_folder = f"{config['datasource']}_{config['model_folder']}"
#     model_filename = f"{config['model_basename']}{epoch}.pt"
#     return str(Path('.') / model_folder / model_filename)

# # Find the latest weights file in the weights folder
# def latest_weights_file_path(config):
#     model_folder = f"{config['datasource']}_{config['model_folder']}"
#     model_filename = f"{config['model_basename']}*"
#     weights_files = list(Path(model_folder).glob(model_filename))
#     if len(weights_files) == 0:
#         return None
#     weights_files.sort()
#     return str(weights_files[-1])
