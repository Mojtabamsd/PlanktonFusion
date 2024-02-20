import datetime
from configs.config import Configuration
from tools.console import Console
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.uvp_dataset import UvpDataset
from models.classifier_cnn import count_parameters
from models.autoencoder import ResNetCustom
import torch
import torch.nn as nn
import torch.optim as optim
from tools.utils import plot_loss
from tools.augmentation import GaussianNoise
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomAffine
import numpy as np
import pandas as pd
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR


# SimCLR framework
class SimCLR(nn.Module):
    def __init__(self, config, encoder_mode=False):
        super(SimCLR, self).__init__()
        self.temperature = config.ssl.temperature
        self.encoder_mode = encoder_mode

        self.encoder = ResNetCustom(num_classes=config.sampling.num_class,
                                    latent_dim=config.ssl.latent_dim,
                                    gray=config.ssl.gray,
                                    pretrained=False)

    def forward(self, x_i, x_j):

        if self.encoder_mode:
            return self.encoder(x_i)

        _, z_i = self.encoder(x_i)
        _, z_j = self.encoder(x_j)

        z_i = nn.functional.normalize(z_i, dim=1)  # Normalize projections
        z_j = nn.functional.normalize(z_j, dim=1)

        return z_i, z_j


def train_ssl(config_path, input_path, output_path):

    config = Configuration(config_path, input_path, output_path)

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    input_folder = input_folder / "train"

    console = Console(output_folder)
    console.info("Training started ...")

    sampled_images_csv_filename = "sampled_images.csv"
    input_csv = input_folder / sampled_images_csv_filename

    if not input_csv.is_file():
        console.error("The input csv file", input_csv, "does not exist.")
        console.quit("Input csv file does not exist.")

    time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    rel_training_path = Path("ssl_training" + time_str)
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
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
        GaussianNoise(std=0.1),
        transforms.ToTensor(),
    ])

    # Create uvp dataset datasets for training and validation
    train_dataset = UvpDataset(root_dir=input_folder,
                               num_class=config.sampling.num_class,
                               csv_file=input_csv,
                               transform=transform,
                               phase='train')

    class_counts = train_dataset.data_frame['label'].value_counts().sort_index().tolist()
    total_samples = sum(class_counts)
    class_weights = [total_samples / (config.sampling.num_class * count) for count in class_counts]
    class_weights_tensor = torch.FloatTensor(class_weights)
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum()

    # Create data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=config.autoencoder.batch_size,
                              shuffle=True)

    device = torch.device(f'cuda:{config.base.gpu_index}' if
                          torch.cuda.is_available() and config.base.cpu is False else 'cpu')
    console.info(f"Running on:  {device}")

    if config.ssl.architecture_type == 'simclr':
        model = SimCLR(config)

    else:
        console.quit("Please select correct parameter for architecture_type")

    # Loss criterion and optimizer
    if config.ssl.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()

    # Calculate the number of parameters in millions
    num_params = count_parameters(model) / 1_000_000
    console.info(f"The model has approximately {num_params:.2f} million parameters.")

    model.to(device)

    # using all available gpu in parallel
    if config.base.all_gpu:
        model = DataParallel(model)
        model = model.cuda()

    # test memory usage
    # console.info(memory_usage(config, model, device))

    # Loss criterion and optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.ssl.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    loss_values = []
    transforms_to_apply = [
        RandomHorizontalFlip(),
        RandomRotation(degrees=15),
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
    ]

    # Training loop
    for epoch in range(config.ssl.num_epoch):
        model.train()
        running_loss = 0.0

        # for images, labels, _ in train_loader:
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            augmented_images = images
            for transform in transforms_to_apply:
                augmented_images = transform(augmented_images)
            images_aug = torch.cat([images, augmented_images], dim=0)

            # # for debug
            # from tools.image import save_img
            # save_img(images_aug, batch_idx, epoch, training_path/"augmented")

            z_i, z_j = model(images_aug[:len(images)], images_aug[len(images):])
            representations = torch.cat([z_i, z_j], dim=0)

            # Compute similarity scores (dot product)
            logits = torch.matmul(representations, representations.T) / model.temperature

            # # Discard diagonals and normalize scores
            # mask = torch.eye(len(representations), device=device)
            # logits = logits - mask * 1e2
            targets = torch.arange(len(representations), device=device)

            optimizer.zero_grad()
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        loss_values.append(average_loss)
        console.info(f"Epoch [{epoch + 1}/{config.ssl.num_epoch}] - Loss: {average_loss:.4f}")
        plot_loss(loss_values, num_epoch=epoch + 1, training_path=config.training_path)

        # save intermediate weight
        if (epoch + 1) % config.ssl.save_model_every_n_epoch == 0:
            # Save the model weights
            saved_weights = f'model_weights_epoch_{epoch + 1}.pth'
            saved_weights_file = training_path / saved_weights

            console.info(f"Model weights saved to {saved_weights_file}")
            torch.save(model.state_dict(), saved_weights_file)

    # Create a plot of the loss values
    plot_loss(loss_values, num_epoch=config.ssl.num_epoch, training_path=config.training_path)

    # Save the model's state dictionary to a file
    saved_weights = "model_weights_final.pth"
    saved_weights_file = training_path / saved_weights

    torch.save(model.state_dict(), saved_weights_file)

    console.info(f"Final model weights saved to {saved_weights_file}")

    # save latent features
    test_dataset = UvpDataset(root_dir=input_folder,
                              num_class=config.sampling.num_class,
                              csv_file=input_csv,
                              transform=transform,
                              phase='test')

    dataloader_test = DataLoader(test_dataset, batch_size=config.classifier.batch_size, shuffle=False)

    all_labels = []
    latent_vectors = []
    model.encoder_mode = True
    with torch.no_grad():
        for index, (images, labels, img_names) in enumerate(dataloader_test):
            images = images.to(device)
            _, latent = model(images, images)

            latent_vectors.extend(latent.cpu().numpy())
            all_labels.append(labels.data.cpu().detach().numpy())

    all_labels = np.concatenate(all_labels).ravel()
    df = pd.DataFrame(latent_vectors, columns=['latent{}'.format(i) for i in range(1, latent_vectors[0].size + 1)])
    int_to_label = {v: k for k, v in dataloader_test.dataset.label_to_int.items()}
    df['labels'] = [int_to_label[label] for label in all_labels]

    report_filename = training_path / 'features.feather'
    df.to_feather(report_filename)



