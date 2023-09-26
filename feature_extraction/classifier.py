import datetime
from configs.config import Configuration
from tools.console import Console
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset.uvp_dataset import UvpDataset
from models.classifier_cnn import count_parameters
from models.autoencoder import ConvAutoencoder
import torch
import pandas as pd
from tools.utils import report_to_df, memory_usage
import os
import shutil
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def classifier(config_path, input_path, output_path):

    config = Configuration(config_path, input_path, output_path)

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    console = Console(output_folder)
    console.info("Classification started ...")

    sampled_images_csv_filename = "sampled_images.csv"
    input_csv = input_folder / sampled_images_csv_filename

    if not input_csv.is_file():
        console.info("Label not provided")
        input_csv = None

    time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    rel_classification_path = Path("classification" + time_str)
    classification_path = output_folder / rel_classification_path
    if not classification_path.exists():
        classification_path.mkdir(exist_ok=True, parents=True)
    elif classification_path.exists():
        console.error("The output folder", classification_path, "exists.")
        console.quit("Folder exists, not overwriting previous results.")

    # Save configuration file
    output_config_filename = classification_path / "config.yaml"
    config.write(output_config_filename)

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((config.sampling.target_size[0], config.sampling.target_size[1])),  # Resize to desired input size
        transforms.ToTensor(),
    ])

    device = torch.device(f'cuda:{config.base.gpu_index}' if
                          torch.cuda.is_available() and config.base.cpu is False else 'cpu')
    console.info(f"Running on:  {device}")

    if config.classifier.feature_type == 'autoencoder':
        model = ConvAutoencoder(latent_dim=config.autoencoder.latent_dim,
                                input_size=config.sampling.target_size,
                                gray=config.autoencoder.gray)

    else:
        console.quit("Please select correct parameter for architecture_type")

    # Calculate the number of parameters in millions
    num_params = count_parameters(model) / 1_000_000
    console.info(f"The model has approximately {num_params:.2f} million parameters.")

    model.to(device)

    # test memory usage
    console.info(memory_usage(config, model, device))

    # Save the model's state dictionary to a file
    saved_weights = "model_weights_final.pth"
    training_path = Path(config.classifier.path_model)
    saved_weights_file = training_path / saved_weights

    console.info("Model loaded from ", saved_weights_file)
    model.load_state_dict(torch.load(saved_weights_file, map_location=device))
    model.to(device)

    test_dataset = UvpDataset(root_dir=input_folder,
                              num_class=config.sampling.num_class,
                              # csv_file=None,
                              csv_file=input_csv,
                              transform=transform,
                              phase='test')

    dataloader = DataLoader(test_dataset, batch_size=config.classifier.batch_size, shuffle=False)

    train_svm(model, dataloader, classification_path, device)


def train_svm(model, dataloader, prediction_path, device):
    model.eval()

    all_labels = []
    latent_vectors = []

    with torch.no_grad():
        for index, (images, labels, img_names) in enumerate(dataloader):
            images = images.to(device)
            outputs, latent = model(images)

            latent_vectors.extend(latent.cpu().numpy())
            all_labels.append(labels.data.cpu().detach().numpy())

        all_labels = np.concatenate(all_labels).ravel()

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(latent_vectors,
                                                            all_labels, test_size=0.2, random_state=42)

        # Train SVM classifier
        svm_classifier = SVC(kernel='rbf')
        svm_classifier.fit(x_train, y_train)

        # Evaluate the SVM classifier
        y_pred = svm_classifier.predict(x_test)

        report = classification_report(
            y_true=y_test,
            y_pred=y_pred,
            target_names=dataloader.dataset.label_to_int,
            digits=6,
        )

        conf_mtx = confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
        )

        df = report_to_df(report)
        report_filename = os.path.join(prediction_path, 'report.csv')
        df.to_csv(report_filename)

        df = pd.DataFrame(conf_mtx)
        conf_mtx_filename = os.path.join(prediction_path, 'conf_matrix.csv')
        df.to_csv(conf_mtx_filename)

    model.train()



