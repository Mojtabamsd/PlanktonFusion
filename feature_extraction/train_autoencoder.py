import datetime
from configs.config import Configuration
from tools.console import Console
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.uvp_dataset import UvpDataset
from models.classifier_cnn import count_parameters
from models.autoencoder import ConvAutoencoder, ResNetCustom, ResNetAutoencoder
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tools.utils import plot_loss, memory_usage
from models.loss import FocalLoss, WeightedCrossEntropyLoss, QuantileLoss, WeightedMSELoss, PerceptualReconstructionLoss
from tools.augmentation import GaussianNoise
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomAffine
import numpy as np
import pandas as pd
from torch.nn.parallel import DataParallel
from tools.visualization import visualization_output
from tools.visualization import tsne_plot


def train_autoencoder(config_path, input_path, output_path):

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

    if config.autoencoder.path_pretrain:
        training_path = Path(config.autoencoder.path_pretrain)
        config.training_path = training_path
    else:
        time_str = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        rel_training_path = Path("autoencoder_training" + time_str)
        training_path = output_folder / rel_training_path
        config.training_path = training_path
        if not training_path.exists():
            training_path.mkdir(exist_ok=True, parents=True)
        elif training_path.exists():
            console.error("The output folder", training_path, "exists.")
            console.quit("Folder exists, not overwriting previous results.")

    visualisation_path = training_path / "visualization"
    if not visualisation_path.exists():
        visualisation_path.mkdir(parents=True, exist_ok=True)

    # Save configuration file
    output_config_filename = training_path / "config.yaml"
    config.write(output_config_filename)

    # # Define data transformations
    # transform = transforms.Compose([
    #     transforms.Resize((config.sampling.target_size[0], config.sampling.target_size[1])),
    #     transforms.ToTensor(),
    # ])

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((config.sampling.target_size[0], config.sampling.target_size[1])),
        RandomHorizontalFlip(),
        RandomRotation(degrees=15),
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

    if config.autoencoder.architecture_type == 'conv_autoencoder':
        model = ConvAutoencoder(latent_dim=config.autoencoder.latent_dim,
                                input_size=config.sampling.target_size,
                                gray=config.autoencoder.gray)

    elif config.autoencoder.architecture_type == 'resnet18':
        model = ResNetCustom(num_classes=config.sampling.num_class,
                             latent_dim=config.autoencoder.latent_dim,
                             gray=config.autoencoder.gray)

    elif config.autoencoder.architecture_type == 'resnet18_autoencoder':
        model = ResNetAutoencoder(latent_dim=config.autoencoder.latent_dim,
                                  input_size=config.sampling.target_size,
                                  gray=config.autoencoder.gray)

    else:
        console.quit("Please select correct parameter for architecture_type")

    if config.autoencoder.path_pretrain:
        pth_files = [file for file in os.listdir(training_path) if
                     file.endswith('.pth') and file != 'model_weights_final.pth']
        epochs = [int(file.split('_')[-1].split('.')[0]) for file in pth_files]
        latest_epoch = max(epochs)
        latest_pth_file = f"model_weights_epoch_{latest_epoch}.pth"

        saved_weights_file = training_path / latest_pth_file

        console.info("Model loaded from ", saved_weights_file)
        model.load_state_dict(torch.load(saved_weights_file, map_location=device))
        model.to(device)
    else:
        latest_epoch = 0

    # Loss criterion and optimizer
    if config.autoencoder.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config.autoencoder.loss == 'cross_entropy_weight':
        class_weights_tensor = class_weights_tensor.to(device)
        criterion = WeightedCrossEntropyLoss(weight=class_weights_tensor)
    elif config.autoencoder.loss == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2)
    elif config.autoencoder.loss == 'mse':
        criterion = nn.MSELoss()
    elif config.autoencoder.loss == 'w_mse':
        class_weights_tensor = class_weights_tensor.to(device)
        criterion = WeightedMSELoss(weight=class_weights_tensor)
    elif config.autoencoder.loss == 'quantile':
        criterion = QuantileLoss(quantile=0.5)
    elif config.autoencoder.loss == 'per_rec':
        criterion = PerceptualReconstructionLoss(config, device)

    # Calculate the number of parameters in millions
    num_params = count_parameters(model) / 1_000_000
    console.info(f"The model has approximately {num_params:.2f} million parameters.")

    model.to(device)

    # using all available gpu in parallel
    if config.base.all_gpu:
        model = DataParallel(model)
        model = model.cuda()

    # test memory usage
    console.info(memory_usage(config, model, device))

    # Loss criterion and optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.autoencoder.learning_rate)

    loss_values = []

    # Training loop
    for epoch in range(latest_epoch, config.autoencoder.num_epoch):

        model.train()
        running_loss = 0.0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            if config.autoencoder.architecture_type == 'conv_autoencoder' or \
                    config.autoencoder.architecture_type == 'resnet18_autoencoder':
                labels_ = labels.clone()
                labels = images

            optimizer.zero_grad()
            outputs, _ = model(images)
            if config.autoencoder.loss == 'w_mse':
                loss = criterion(outputs, labels, labels_)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        loss_values.append(average_loss)
        console.info(f"Epoch [{epoch + 1}/{config.autoencoder.num_epoch}] - Loss: {average_loss:.4f}")
        plot_loss(loss_values, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path)

        # visualize every 5 epoch
        if (epoch + 1) % 5 == 0:
            visualization_output(images, outputs, visualisation_path,
                                 epoch, batch_size=config.autoencoder.batch_size, gray=config.autoencoder.gray)

        # save intermediate weight
        if (epoch + 1) % config.autoencoder.save_model_every_n_epoch == 0:
            # Save the model weights
            saved_weights = f'model_weights_epoch_{epoch + 1}.pth'
            saved_weights_file = training_path / saved_weights

            console.info(f"Model weights saved to {saved_weights_file}")
            torch.save(model.state_dict(), saved_weights_file)

    # Create a plot of the loss values
    plot_loss(loss_values, num_epoch=(config.autoencoder.num_epoch - latest_epoch), training_path=config.training_path)

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
    with torch.no_grad():
        for index, (images, labels, img_names) in enumerate(dataloader_test):
            images = images.to(device)
            _, latent = model(images)

            latent_vectors.extend(latent.cpu().numpy())

            # # for visualize the tsne of whole image
            # latent_vectors.extend(np.reshape(images.cpu().numpy(),
            #                                  (images.shape[0],images.shape[1]*images.shape[2]*images.shape[3])))

            all_labels.append(labels.data.cpu().detach().numpy())

    all_labels = np.concatenate(all_labels).ravel()
    df = pd.DataFrame(latent_vectors, columns=['latent{}'.format(i) for i in range(1, latent_vectors[0].size + 1)])
    int_to_label = {v: k for k, v in dataloader_test.dataset.label_to_int.items()}
    df['labels'] = [int_to_label[label] for label in all_labels]

    report_filename = training_path / 'features.feather'
    df.to_feather(report_filename)

    # tsne plot
    tsne_plot(latent_vectors, all_labels, int_to_label, visualisation_path)



