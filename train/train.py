import pandas as pd
import numpy as np
from pathlib import Path
import datetime
import sys
from configs.config import Configuration
from tools.Console import Console
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset.uvp_dataset import UvpDataset
from models.architecture import SimpleCNN
import torch


def train_cnn(config_path, input_path, output_path):

    Console.info("Sampling started at", datetime.datetime.now())
    config = Configuration(config_path, input_path, output_path)

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    sampled_images_csv_filename = "sampled_images.csv"
    input_csv = input_folder / sampled_images_csv_filename

    if not input_csv.is_file():
        Console.error("The input csv file", input_csv, "does not exist.")
        Console.quit("Input csv file does not exist.")

    time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    rel_training_path = Path("training" + time_str)
    training_path = output_folder / rel_training_path
    if not training_path.exists():
        training_path.mkdir(exist_ok=True, parents=True)
    elif training_path.exists():
        Console.error("The output folder", training_path, "exists.")
        Console.quit("Folder exists, not overwriting previous results.")

    # Save configuration file
    output_config_filename = training_path / "config.yaml"
    config.write(output_config_filename)

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to desired input size
        transforms.ToTensor(),
    ])

    # Create uvp dataset
    dataset = UvpDataset(csv_file=input_csv, root_dir=input_folder, transform=transform)

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)

    config.num_class = dataset.data_frame['label'].nunique()
    device = torch.device(f'cuda:{config.base.gpu_index}' if
                          torch.cuda.is_available() and config.base.cpu is False else 'cpu')

    model = SimpleCNN(config.num_class, gray=config.training.gray)

    # training loop
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Your training code here
        pass

    a=1





