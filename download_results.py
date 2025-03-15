import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv(override=True)


def authenticate(kaggle_config_path: Optional[str] = None) -> None:
    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        error_msg = "Kaggle API credentials not found in environment variables"
        print(error_msg)
        raise RuntimeError(error_msg)

    print("Using Kaggle API credentials from environment variables")

    kaggle_config = {
        "username": kaggle_username,
        "key": kaggle_key
    }

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    config_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(config_path, "w") as f:
        json.dump(kaggle_config, f)

    os.chmod(config_path, 0o600)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("Kaggle API authentication successful")
        return api
    except Exception as e:
        error_msg = f"Kaggle API authentication failed: {e}"
        print(error_msg)
        raise RuntimeError(error_msg)


def download_kernels_outputs():
    api = authenticate()

    kernels = [
        "sayedgamal99/advanced-face-module-facenet512",
        "sayedgamal99/advanced-face-module-arcface"
    ]

    output_dirs = [
        "notebooks_outputs/facenet",
        "notebooks_outputs/arcface"
    ]

    for kernel, output_dir in zip(kernels, output_dirs):
        print(f"Downloading output of kernel {kernel} into {output_dir}")
        api.kernels_output(kernel, path=output_dir)
        print(f"Downloaded output of kernel {kernel} into {output_dir}")


if __name__ == "__main__":
    download_kernels_outputs()
