import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from train.load_datasets import load_datasets
from train.train_model import train_model

def train():
    dataset_path = 'ibsr_3d/'

    train_ds, val_ds = load_datasets(dataset_path)
    train_model(train_ds, val_ds)
    
if __name__ == "__main__":
    train()