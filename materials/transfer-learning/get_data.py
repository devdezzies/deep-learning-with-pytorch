"""
contains data setup functionality
"""
from torchvision import datasets 
from torch.utils.data import DataLoader
from pathlib import Path 
from zipfile import ZipFile
import requests
import os

DATA_PATH = Path("data")
DATA_PATH.mkdir(parents=True, exist_ok=True)

IMAGE_FILE = "pizza_sushi_steak.zip"
IMAGE_DIR = DATA_PATH / IMAGE_FILE
if IMAGE_DIR.is_file():
    print("image datasets already exist")
else:
    print("creating dataset directory..")
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip")
    with open(DATA_PATH / IMAGE_FILE, "wb") as f: 
        f.write(request.content)

FILE_PATH = DATA_PATH / "pizza_sushi_steak"
if FILE_PATH.is_dir():
    print("file already extracted")
else:
    print("extracting..")
    with ZipFile(IMAGE_DIR, "r") as zip_ref:
        zip_ref.extractall(FILE_PATH)
    os.remove(IMAGE_DIR)
