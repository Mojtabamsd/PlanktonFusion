import datetime
from configs.config import Configuration
from tools.console import Console
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset.uvp_dataset import UvpDataset
from models.classifier_cnn import ResNetCustom
from models.autoencoder import ConvAutoencoder, ResNetAutoencoder
import torch
import pandas as pd
from tools.utils import report_to_df
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def prediction_auto(config_path, input_path, output_path):

    config = Configuration(config_path, input_path, output_path)

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    console = Console(output_folder)
    console.info("Prediction started ...")

    sampled_images_csv_filename = "sampled_images.csv"
    input_csv = input_folder / sampled_images_csv_filename

    if not input_csv.is_file():
        console.info("Label not provided")
        input_csv = None

    time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    rel_prediction_path = Path("prediction_auto" + time_str)
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
        transforms.Resize((config.sampling.target_size[0], config.sampling.target_size[1])),  # Resize to desired input size
        transforms.ToTensor(),
    ])

    device = torch.device(f'cuda:{config.base.gpu_index}' if
                          torch.cuda.is_available() and config.base.cpu is False else 'cpu')
    console.info(f"Running on:  {device}")

    if config.prediction_auto.architecture_type1 == 'conv_autoencoder':
        model1 = ConvAutoencoder(latent_dim=config.prediction_auto.latent_dim,
                                 input_size=config.sampling.target_size,
                                 gray=config.training.gray)

    elif config.prediction_auto.architecture_type1 == 'resnet18_autoencoder':
        model1 = ResNetAutoencoder(latent_dim=config.prediction_auto.latent_dim,
                                   input_size=config.sampling.target_size,
                                   gray=config.training.gray)
    else:
        console.quit("Please select correct parameter for architecture_type1")

    if config.prediction_auto.architecture_type2 == 'resnet18':
        model2 = ResNetCustom(num_classes=config.sampling.num_class,
                              input_size=config.sampling.target_size,
                              gray=config.training.gray)



    else:
        console.quit("Please select correct parameter for architecture_type2")

    model1.to(device)
    model2.to(device)

    # Save the model's state dictionary to a file
    saved_weights = "model_weights_final.pth"
    training_path1 = Path(config.prediction_auto.path_model_auto)
    saved_weights_file1 = training_path1 / saved_weights

    training_path2 = Path(config.prediction_auto.path_model_class)
    saved_weights_file2 = training_path2 / saved_weights

    console.info("Model 1 loaded from ", saved_weights_file1)
    model1.load_state_dict(torch.load(saved_weights_file1, map_location=device))
    model1.to(device)

    console.info("Model 2 loaded from ", saved_weights_file2)
    model2.load_state_dict(torch.load(saved_weights_file2, map_location=device))
    model2.to(device)

    test_dataset = UvpDataset(root_dir=input_folder,
                              num_class=config.sampling.num_class,
                              # csv_file=None,
                              csv_file=input_csv,
                              transform=transform,
                              phase='test')

    dataloader = DataLoader(test_dataset, batch_size=config.prediction_auto.batch_size, shuffle=False)

    predict(model1, model2, dataloader, prediction_path, device)


def predict(model1, model2, dataloader, prediction_path, device):
    model1.eval()
    model2.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for index, (images, labels, img_names) in enumerate(dataloader):
            images = images.to(device)
            reconstructed, _ = model1(images)
            outputs = model2(reconstructed)
            _, predicted_labels = torch.max(outputs, 1)

            all_labels.append(labels.data.cpu().detach().numpy())
            all_preds.append(predicted_labels.cpu().detach().numpy())

            print('batch ' + str(index) + '     out of:     ' + str(dataloader.__len__()))

            for i in range(len(predicted_labels)):
                int_label = predicted_labels[i].item()
                string_label = dataloader.dataset.get_string_label(int_label)
                image_name = img_names[i]
                image_path = os.path.join(prediction_path, 'output', string_label, image_name.replace('output/', ''))

                if not os.path.exists(os.path.dirname(image_path)):
                    os.makedirs(os.path.dirname(image_path))

                # input_path = os.path.join(dataloader.dataset.root_dir, image_name)
                pil_image = transforms.ToPILImage()(reconstructed[i, :])
                pil_image.save(image_path)

        all_labels = np.concatenate(all_labels).ravel()
        all_preds = np.concatenate(all_preds).ravel()

        report = classification_report(
            all_labels,
            all_preds,
            target_names=dataloader.dataset.label_to_int,
            digits=6,
        )

        conf_mtx = confusion_matrix(
            all_labels,
            all_preds,
        )

        df = report_to_df(report)
        report_filename = os.path.join(prediction_path, 'report.csv')
        df.to_csv(report_filename)

        df = pd.DataFrame(conf_mtx)
        conf_mtx_filename = os.path.join(prediction_path, 'conf_matrix.csv')
        df.to_csv(conf_mtx_filename)



