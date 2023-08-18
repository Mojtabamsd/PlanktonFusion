import datetime
import shutil

from configs.config import Configuration
from tools.console import Console
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset.uvp_dataset import UvpDataset
from models.architecture import SimpleCNN
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os
from PIL import Image
import shutil


def prediction(config_path, input_path, output_path):

    console = Console()
    console.info("Prediction started ...")

    config = Configuration(config_path, input_path, output_path)

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    sampled_images_csv_filename = "sampled_images.csv"
    input_csv = input_folder / sampled_images_csv_filename

    if not input_csv.is_file():
        console.info("Label not provided")
        input_csv = None

    time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    rel_prediction_path = Path("prediction" + time_str)
    prediction_path = output_folder / rel_prediction_path
    if not prediction_path.exists():
        prediction_path.mkdir(exist_ok=True, parents=True)
    elif prediction_path.exists():
        console.error("The output folder", prediction_path, "exists.")
        console.quit("Folder exists, not overwriting previous results.")

    # Save configuration file
    output_config_filename = prediction_path / "config.yaml"
    config.write(output_config_filename)

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((227, 227)),  # Resize to desired input size
        transforms.ToTensor(),
    ])

    device = torch.device(f'cuda:{config.base.gpu_index}' if
                          torch.cuda.is_available() and config.base.cpu is False else 'cpu')
    model = SimpleCNN(num_classes=config.sampling.num_class,
                      gray=config.training.gray,
                      input_size=config.sampling.target_size)
    model.to(device)

    # Save the model's state dictionary to a file
    saved_weights = "model_weights.pth"
    training_path = Path(config.prediction.path_model)
    saved_weights_file = training_path / saved_weights

    console.info("Model loaded from ", saved_weights_file)
    model.load_state_dict(torch.load(saved_weights_file, map_location=device))
    model.to(device)

    test_dataset = UvpDataset(root_dir=input_folder,
                              num_class=config.sampling.num_class,
                              # csv_file=None,
                              csv_file=input_csv,
                              transform=transform,
                              train=False)

    dataloader = DataLoader(test_dataset, batch_size=config.prediction.batch_size, shuffle=False)

    predict(model, dataloader, prediction_path, device)


def predict(model, dataloader, prediction_path, device):
    model.eval()

    with torch.no_grad():
        for index, (images, _, img_names) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            _, predicted_labels = torch.max(outputs, 1)

            for i in range(len(predicted_labels)):
                int_label = predicted_labels[i].item()
                string_label = dataloader.dataset.get_string_label(int_label)
                image_name = img_names[i]
                image_path = os.path.join(prediction_path, string_label, image_name.replace('output/', ''))

                if not os.path.exists(os.path.dirname(image_path)):
                    os.makedirs(os.path.dirname(image_path))

                input_path = os.path.join(dataloader.dataset.root_dir, image_name)
                shutil.copy(input_path, image_path)

    model.train()


