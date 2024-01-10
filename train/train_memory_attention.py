import datetime
from configs.config import Configuration
from tools.console import Console
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset.uvp_dataset import UvpDataset
from models.classifier_cnn import count_parameters
from models.autoencoder import ConvAutoencoder, ResNetCustom
import torch
import torch.nn as nn
import torch.optim as optim
from tools.utils import report_to_df, plot_loss, memory_usage
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomAffine
from tools.augmentation import GaussianNoise
from models.loss import FocalLoss, WeightedCrossEntropyLoss, LogitAdjustmentLoss
from models.memory_attention import MA


def train_memory(config_path, input_path, output_path):

    config = Configuration(config_path, input_path, output_path)
    phase = 'train'      # will train with whole dataset and testing results if there is a test file
    # phase = 'train_val'  # will train with 80% dataset and testing results with the rest 20% of data

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    input_folder_train = input_folder / "train"
    input_folder_test = input_folder / "test"

    console = Console(output_folder)
    console.info("Training started ...")

    sampled_images_csv_filename = "sampled_images.csv"
    input_csv_train = input_folder_train / sampled_images_csv_filename
    input_csv_test = input_folder_test / sampled_images_csv_filename

    if not input_csv_train.is_file():
        console.info("Label not provided for training")
        input_csv_train = None

    if not input_csv_test.is_file():
        console.info("Label not provided for testing")
        input_csv_test = None

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
        RandomHorizontalFlip(),
        RandomRotation(degrees=15),
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
        GaussianNoise(std=0.1),
        transforms.ToTensor(),
    ])

    # Create uvp dataset datasets for training and validation
    train_dataset = UvpDataset(root_dir=input_folder_train,
                               num_class=config.sampling.num_class,
                               csv_file=input_csv_train,
                               transform=transform,
                               phase=phase)

    class_counts = train_dataset.data_frame['label'].value_counts().sort_index().tolist()
    total_samples = sum(class_counts)
    class_weights = [total_samples / (config.sampling.num_class * count) for count in class_counts]
    class_weights_tensor = torch.FloatTensor(class_weights)
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum()

    class_balancing = False
    if class_balancing:
        # class balanced sampling
        sampler = torch.utils.data.WeightedRandomSampler(weights=class_weights_tensor,
                                                         num_samples=len(train_dataset),
                                                         replacement=True)
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.training.batch_size,
                                  sampler=sampler)

    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.training.batch_size,
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

    else:
        console.quit("Please select correct parameter for architecture_type")

    # Calculate the number of parameters in millions
    num_params = count_parameters(model) / 1_000_000
    console.info(f"The model has approximately {num_params:.2f} million parameters.")

    model.to(device)

    # test memory usage
    console.info(memory_usage(config, model, device))

    # Loss criterion and optimizer
    if config.memory.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config.training.loss == 'cross_entropy_weight':
        class_weights_tensor = class_weights_tensor.to(device)
        criterion = WeightedCrossEntropyLoss(weight=class_weights_tensor)
    elif config.training.loss == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2)
    elif config.training.loss == 'LACE':
        class_weights_tensor = class_weights_tensor.to(device)
        criterion = LogitAdjustmentLoss(weight=class_weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    loss_values = []

    # load visual embedding and attention model

    model = MA(model_name=config.autoencoder.architecture_type,
               weights_path=config.memory.path_model,
               visual_encoder_size=config.autoencoder.latent_dim,
               query_size=config.autoencoder.latent_dim,
               memory_size=config.autoencoder.latent_dim,
               attention_units=256,
               num_classes=config.sampling.num_class,
               k=config.memory.k)






