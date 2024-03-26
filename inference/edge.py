import datetime
from configs.config import Configuration
from tools.console import Console
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset.uvp_dataset import UvpDataset
from models.classifier_cnn import SimpleCNN, ResNetCustom, MobileNetCustom, ShuffleNetCustom, count_parameters
import torch
import pandas as pd
from tools.utils import report_to_df, memory_usage, processing_time
import os, json, shutil
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def edge(config_path, input_path, output_path):

    config = Configuration(config_path, input_path, output_path)

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    console = Console(output_folder)
    console.info("Model loading started ...")

    sampled_images_csv_filename = "sampled_images.csv"
    input_csv = input_folder / sampled_images_csv_filename

    if not input_csv.is_file():
        console.info("Label not provided")
        input_csv = None

    time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    rel_prediction_path = Path("prediction" + time_str)
    prediction_path = output_folder / rel_prediction_path
    config.prediction_path = prediction_path
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

    if config.training.architecture_type == 'simple_cnn':
        model = SimpleCNN(num_classes=config.sampling.num_class,
                          input_size=config.sampling.target_size,
                          gray=config.training.gray)

    elif config.training.architecture_type == 'resnet18':
        model = ResNetCustom(num_classes=config.sampling.num_class,
                             input_size=config.sampling.target_size,
                             gray=config.training.gray)

    elif config.training.architecture_type == 'mobilenet':
        model = MobileNetCustom(num_classes=config.sampling.num_class,
                                input_size=config.sampling.target_size,
                                gray=config.training.gray)

    elif config.training.architecture_type == 'shufflenet':
        model = ShuffleNetCustom(num_classes=config.sampling.num_class,
                                 input_size=config.sampling.target_size,
                                 gray=config.training.gray)

    else:
        console.quit("Please select correct parameter for architecture_type")

    # Calculate the number of parameters in millions
    num_params = count_parameters(model) / 1_000_000
    console.info(f"The model has approximately {num_params:.2f} million parameters.")

    model.to(device)

    # test memory and time usage
    console.info(memory_usage(config, model, device))
    console.info(processing_time(config, model, device))

    # Save the model's state dictionary to a file
    saved_weights = "model_weights_final.pth"
    training_path = Path(config.prediction.path_model)
    saved_weights_file = training_path / saved_weights

    console.info("Model loaded from ", saved_weights_file)
    model.load_state_dict(torch.load(saved_weights_file, map_location=device))
    model.to(device)

    permitted_formats = ['.bmp', '.jpeg', '.png']
    test_dataset = UvpDataset(root_dir=input_folder,
                              num_class=config.sampling.num_class,
                              # csv_file=None,
                              csv_file=input_csv,
                              transform=transform,
                              phase='test',
                              permitted_formats=permitted_formats)

    dataloader = DataLoader(test_dataset, batch_size=config.prediction.batch_size, shuffle=False)

    predict_edge(model, dataloader, device, console, config)


def predict_edge(model, dataloader, device, console, config):
    console.info("Edge processing started ...")
    model.eval()

    all_labels = []
    all_preds = []
    classification_summary = []
    total_images_processed = 0

    with torch.no_grad():
        for index, (images, labels, img_names) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            _, predicted_labels = torch.max(outputs, 1)

            if dataloader.dataset.csv_file:
                all_labels.append(labels.data.cpu().detach().numpy())
            all_preds.append(predicted_labels.cpu().detach().numpy())

            for i in range(len(predicted_labels)):
                int_label = predicted_labels[i].item()
                string_label = dataloader.dataset.get_string_label(int_label)
                image_name = img_names[i]
                image_path = os.path.join(config.prediction_path, string_label, image_name.replace('output/', ''))
                input_path = os.path.join(dataloader.dataset.root_dir, image_name)

                if config.edge.save_image:
                    if not os.path.exists(os.path.dirname(image_path)):
                        os.makedirs(os.path.dirname(image_path))
                    shutil.copy(input_path, image_path)

                log = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "image_name": image_name,
                       "predicted_class": string_label,
                       "image_path": input_path}
                classification_summary.append(log)

                total_images_processed += 1

                if total_images_processed % 1000 == 0:
                    console.info(f"Total number of images processed: {total_images_processed}")

        console.info(f"Total number of images processed: {total_images_processed}")
        if dataloader.dataset.csv_file:
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
            report_filename = os.path.join(config.prediction_path, 'report.csv')
            df.to_csv(report_filename)

            df = pd.DataFrame(conf_mtx)
            conf_mtx_filename = os.path.join(config.prediction_path, 'conf_matrix.csv')
            df.to_csv(conf_mtx_filename)

    # summary reporting file
    summary_filename = config.prediction_path / "summary.json"
    with open(summary_filename, "w") as f:
        json.dump(classification_summary, f, indent=4)

