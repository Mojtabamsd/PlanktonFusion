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
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomAffine
from tools.augmentation import RandomZoomIn, RandomZoomOut, GaussianNoise
from models.loss import FocalLoss, WeightedCrossEntropyLoss



def train_cnn(config_path, input_path, output_path):

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
        # RandomZoomIn(zoom_range=(0.8, 1.0)),
        # RandomZoomOut(zoom_range=(1.0, 1.2)),
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

    oversampling = False
    if oversampling:
        # oversampling the minority classes
        sampler = torch.utils.data.WeightedRandomSampler(weights=class_weights_tensor,
                                                         num_samples=len(train_dataset), replacement=True)
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
    if config.training.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config.training.loss == 'cross_entropy_weight':
        class_weights_tensor = class_weights_tensor.to(device)
        criterion = WeightedCrossEntropyLoss(weight=class_weights_tensor)
    elif config.training.loss == 'focal':
        criterion = FocalLoss(alpha=1, gamma=2)

    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    loss_values = []

    # Training loop
    for epoch in range(config.training.num_epoch):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # # for debug
            # from tools.image import save_img
            # save_img(images, batch_idx, epoch, training_path/"augmented")

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
    plot_loss(loss_values, num_epoch=config.training.num_epoch, training_path=config.training_path)

    # Save the model's state dictionary to a file
    saved_weights = "model_weights_final.pth"
    saved_weights_file = training_path / saved_weights

    torch.save(model.state_dict(), saved_weights_file)

    console.info(f"Final model weights saved to {saved_weights_file}")

    # Create uvp dataset datasets for training and validation
    if phase == 'train_val':
        console.info('Testing model with validation subset')
        train_dataset.phase = 'val'
        val_dataset = train_dataset

        val_loader = DataLoader(val_dataset,
                                batch_size=config.training.batch_size,
                                shuffle=True)

    elif input_csv_test is not None:
        console.info('Testing model with folder test')

        test_dataset = UvpDataset(root_dir=input_folder_test,
                                  num_class=config.sampling.num_class,
                                  csv_file=input_csv_test,
                                  transform=transform,
                                  phase='test')

        val_loader = DataLoader(test_dataset,
                                batch_size=config.classifier.batch_size,
                                shuffle=True)
    else:
        console.quit('no data for testing model')

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





