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


def feature_uvpec(config_path, input_path, output_path):

    config = Configuration(config_path, input_path, output_path)

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    console = Console(output_folder)
    console.info("Feature Extraction started ...")

    sampled_images_csv_filename = "sampled_images.csv"
    input_csv = input_folder / sampled_images_csv_filename

    if not input_csv.is_file():
        console.info("Label not provided")
        input_csv = None

    time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    rel_feature_path = Path("feature" + time_str)
    feature_path = output_folder / rel_feature_path
    if not feature_path.exists():
        feature_path.mkdir(exist_ok=True, parents=True)
    elif feature_path.exists():
        console.error("The output folder", feature_path, "exists.")
        console.quit("Folder exists, not overwriting previous results.")

    # Save configuration file
    output_config_filename = feature_path / "config.yaml"
    config.write(output_config_filename)

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((config.sampling.target_size[0], config.sampling.target_size[1])),  # Resize to desired input size
        transforms.ToTensor(),
    ])

    device = torch.device(f'cuda:{config.base.gpu_index}' if
                          torch.cuda.is_available() and config.base.cpu is False else 'cpu')
    console.info(f"Running on:  {device}")

    if config.feature_uvpec.feature_type == 'conv_autoencoder':
        model = ConvAutoencoder(latent_dim=config.autoencoder.latent_dim,
                                input_size=config.sampling.target_size,
                                gray=config.autoencoder.gray)

    else:
        console.quit("Please select correct parameter for feature_type")

    # Calculate the number of parameters in millions
    num_params = count_parameters(model) / 1_000_000
    console.info(f"The model has approximately {num_params:.2f} million parameters.")

    model.to(device)

    # test memory usage
    console.info(memory_usage(config, model, device))

    # Save the model's state dictionary to a file
    saved_weights = "model_weights_final.pth"
    training_path = Path(config.feature_uvpec.path_model)
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

    sub_folder = input_path + r'\output'
    latent_extraction(model, dataloader, feature_path, sub_folder, device)


def latent_extraction(model, dataloader, prediction_path, sub_folder, device):
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

        df = pd.DataFrame(latent_vectors, columns=['latent{}'.format(i) for i in range(1, latent_vectors[0].size+1)])
        int_to_label = {v: k for k, v in dataloader.dataset.label_to_int.items()}
        df['labels'] = [int_to_label[label] for label in all_labels]

        report_filename = os.path.join(prediction_path, 'features.feather')
        df.to_feather(report_filename)

        # split taxon names and their IDs from EcoTaxa

        _, folder_list, _ = next(os.walk(sub_folder))
        folder_list_split = [folder.split('_') for folder in folder_list]
        dico_id = {folder_list_split[i][0]: folder_list_split[i][1] for i in range(len(folder_list_split))}
        dic_path = os.path.join(prediction_path, 'dico_id.npy')
        np.save(dic_path, dico_id)



