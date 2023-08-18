import datetime
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


def train_cnn(config_path, input_path, output_path):

    console = Console()
    console.info("Training started ...")

    config = Configuration(config_path, input_path, output_path)

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    sampled_images_csv_filename = "sampled_images.csv"
    input_csv = input_folder / sampled_images_csv_filename

    if not input_csv.is_file():
        console.error("The input csv file", input_csv, "does not exist.")
        console.quit("Input csv file does not exist.")

    time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    rel_training_path = Path("training" + time_str)
    training_path = output_folder / rel_training_path
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
        transforms.Resize((227, 227)),  # Resize to desired input size
        transforms.ToTensor(),
    ])

    # Create uvp dataset datasets for training and validation
    train_dataset = UvpDataset(root_dir=input_folder,
                               num_class=config.sampling.num_class,
                               csv_file=input_csv,
                               transform=transform,
                               train=True)

    val_dataset = UvpDataset(root_dir=input_folder,
                             num_class=config.sampling.num_class,
                             csv_file=input_csv,
                             transform=transform,
                             train=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=config.training.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=config.training.batch_size,
                            shuffle=False)

    device = torch.device(f'cuda:{config.base.gpu_index}' if
                          torch.cuda.is_available() and config.base.cpu is False else 'cpu')

    model = SimpleCNN(num_classes=config.sampling.num_class,
                      gray=config.training.gray,
                      input_size=config.sampling.target_size)
    model.to(device)

    # Loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Training loop
    for epoch in range(config.training.num_epoch):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        console.info(f"Epoch [{epoch + 1}/{config.training.num_epoch}] - Loss: {average_loss:.4f}")

    # Save the model's state dictionary to a file
    saved_weights = "model_weights.pth"
    saved_weights_file = training_path / saved_weights

    torch.save(model.state_dict(), saved_weights_file)

    console.info(f"Model weights saved to {saved_weights_file}")

    # Evaluation loop
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    console.info(f"Validation Accuracy: {accuracy:.4f}")

    console.save_log(training_path)





