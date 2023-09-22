import datetime
from configs.config import Configuration
from tools.console import Console
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset.uvp_dataset import UvpDataset
from models.classifier_cnn import SimpleCNN, ResNetCustom, MobileNetCustom, ShuffleNetCustom, count_parameters
import torch
import torch.nn as nn
import torch.optim as optim
from tools.utils import report_to_df, plot_loss, memory_usage
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def train_cnn(config_path, input_path, output_path):

    config = Configuration(config_path, input_path, output_path)

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    console = Console(output_folder)
    console.info("Training started ...")

    sampled_images_csv_filename = "sampled_images.csv"
    input_csv = input_folder / sampled_images_csv_filename

    if not input_csv.is_file():
        console.error("The input csv file", input_csv, "does not exist.")
        console.quit("Input csv file does not exist.")

    time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    rel_training_path = Path("training" + time_str)
    training_path = output_folder / rel_training_path
    config.training_path = training_path
    if not training_path.exists():
        training_path.mkdir(exist_ok=True, parents=True)
    elif training_path.exists():
        console.error("The output folder", training_path, "exists.")
        console.quit("Folder exists, not overwriting previous results.")

    # Save configuration file
    output_config_filename = training_path / "config.yaml"
    config.write(output_config_filename)

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((config.sampling.target_size[0], config.sampling.target_size[1])),
        transforms.ToTensor(),
    ])

    # Create uvp dataset datasets for training and validation
    train_dataset = UvpDataset(root_dir=input_folder,
                               num_class=config.sampling.num_class,
                               csv_file=input_csv,
                               transform=transform,
                               phase='train')

    # Create data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=config.training.batch_size,
                              shuffle=True)

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

    # test memory usage
    console.info(memory_usage(config, model, device))

    # Loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    loss_values = []

    # Training loop
    for epoch in range(config.training.num_epoch):
        model.train()
        running_loss = 0.0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        loss_values.append(average_loss)
        console.info(f"Epoch [{epoch + 1}/{config.training.num_epoch}] - Loss: {average_loss:.4f}")

        # save intermediate weight
        if (epoch + 1) % config.training.save_model_every_n_epoch == 0:
            # Save the model weights
            saved_weights = f'model_weights_epoch_{epoch + 1}.pth'
            saved_weights_file = training_path / saved_weights

            console.info(f"Model weights saved to {saved_weights_file}")
            torch.save(model.state_dict(), saved_weights_file)

    # Create a plot of the loss values
    plot_loss(loss_values, config)

    # Save the model's state dictionary to a file
    saved_weights = "model_weights_final.pth"
    saved_weights_file = training_path / saved_weights

    torch.save(model.state_dict(), saved_weights_file)

    console.info(f"Final model weights saved to {saved_weights_file}")

    # Create uvp dataset datasets for training and validation
    train_dataset.phase = 'val'
    val_dataset = train_dataset

    # Create data loaders
    val_loader = DataLoader(val_dataset,
                            batch_size=config.training.batch_size,
                            shuffle=True)

    # Evaluation loop
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=train_dataset.label_to_int,
        digits=6,
    )

    conf_mtx = confusion_matrix(
        all_labels,
        all_preds,
    )

    df = report_to_df(report)
    report_filename = training_path / 'report_evaluation.csv'
    df.to_csv(report_filename)

    df = pd.DataFrame(conf_mtx)
    conf_mtx_filename = training_path / 'conf_matrix_evaluation.csv'
    df.to_csv(conf_mtx_filename)

    console.info('************* Evaluation Report *************')
    console.info(report)
    console.save_log(training_path)





