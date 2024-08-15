import datetime
from configs.config import Configuration
from tools.console import Console
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset.uvp_dataset import UvpDataset
from models.classifier_cnn import SimpleCNN, ResNetCustom, MobileNetCustom, ShuffleNetCustom, count_parameters
from models import resnext
from models.classifier_vit import ViT, ViTPretrained
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from tools.utils import report_to_df, plot_loss, memory_usage
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomAffine, RandomResizedCrop, \
    ColorJitter, RandomGrayscale, RandomPerspective, RandomVerticalFlip
from tools.augmentation import GaussianNoise, ResizeAndPad
from models.loss import FocalLoss, WeightedCrossEntropyLoss, LogitAdjustmentLoss, LogitAdjust
from transformers import ViTForImageClassification
from models.proco import ProCoLoss


def train_nn(config_path, input_path, output_path):

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

    config.input_csv_train = str(input_csv_train)
    config.input_csv_test = str(input_csv_test)

    if not input_csv_train.is_file():
        console.info("Label not provided for training")
        input_csv_train = None

    if not input_csv_test.is_file():
        console.info("Label not provided for testing")
        input_csv_test = None

    if config.training.path_pretrain:
        training_path = Path(config.training.path_pretrain)
        config.training_path = training_path
    else:
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
    if config.training.padding:
        transform = transforms.Compose([
            ResizeAndPad((config.training.target_size[0], config.training.target_size[1])),
            RandomHorizontalFlip(),
            RandomRotation(degrees=30),
            RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
            GaussianNoise(std=0.1),
            RandomResizedCrop((config.training.target_size[0], config.training.target_size[1])),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            RandomGrayscale(p=0.1),
            RandomPerspective(distortion_scale=0.2, p=0.5),
            RandomVerticalFlip(p=0.1),
            transforms.ToTensor(),
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize((config.training.target_size[0], config.training.target_size[1])),
            RandomHorizontalFlip(),
            RandomRotation(degrees=30),
            RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
            GaussianNoise(std=0.1),
            RandomResizedCrop((config.training.target_size[0], config.training.target_size[1])),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            RandomGrayscale(p=0.1),
            RandomPerspective(distortion_scale=0.2, p=0.5),
            RandomVerticalFlip(p=0.1),
            transforms.ToTensor(),
        ])

    # Create uvp dataset datasets for training and validation
    train_dataset = UvpDataset(root_dir=input_folder_train,
                               num_class=config.sampling.num_class,
                               csv_file=input_csv_train,
                               transform=transform,
                               phase=phase,
                               gray=config.training.gray)

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
                                  shuffle=True,
                                  num_workers=config.training.num_workers)

    device = torch.device(f'cuda:{config.base.gpu_index}' if
                          torch.cuda.is_available() and config.base.cpu is False else 'cpu')
    console.info(f"Running on:  {device}")

    model = resnext.Model(name=config.training.architecture_type,
                          num_classes=config.sampling.num_class,
                          feat_dim=512,
                          use_norm=False,
                          gray=config.training.gray)

    # if config.training.architecture_type == 'simple_cnn':
    #     model = SimpleCNN(num_classes=config.sampling.num_class,
    #                       input_size=config.training.target_size,
    #                       gray=config.training.gray)
    #
    # elif config.training.architecture_type == 'resnet18':
    #     model = ResNetCustom(num_classes=config.sampling.num_class,
    #                          input_size=config.training.target_size,
    #                          gray=config.training.gray,
    #                          pretrained=config.training.pre_train,
    #                          freeze_layers=False)
    #
    # elif config.training.architecture_type == 'mobilenet':
    #     model = MobileNetCustom(num_classes=config.sampling.num_class,
    #                             input_size=config.training.target_size,
    #                             gray=config.training.gray)
    #
    # elif config.training.architecture_type == 'shufflenet':
    #     model = ShuffleNetCustom(num_classes=config.sampling.num_class,
    #                              input_size=config.training.target_size,
    #                              gray=config.training.gray)
    #
    # elif config.training.architecture_type == 'vit_base':
    #     model = ViT(input_size=config.training.target_size[0], patch_size=16, num_classes=config.sampling.num_class,
    #                 dim=256, depth=12, heads=8, mlp_dim=512, gray=config.training.gray, dropout=0.1)
    #
    # elif config.training.architecture_type == 'vit_pretrained':
    #     # pretrained_model_name = "vit_base_patch16_224"
    #     # model = ViTPretrained(pretrained_model_name, num_classes=config.sampling.num_class, gray=config.training.gray)
    #     model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224",
    #                                                       num_labels=config.sampling.num_class,
    #                                                       ignore_mismatched_sizes=True)
    #
    # else:
    #     console.quit("Please select correct parameter for architecture_type")

    # Calculate the number of parameters in millions
    num_params = count_parameters(model) / 1_000_000
    console.info(f"The model has approximately {num_params:.2f} million parameters.")

    model.to(device)

    # test memory usage
    # console.info(memory_usage(config, model, device))

    if config.training.path_pretrain:
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
    if config.training.loss == 'cross_entropy':
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

    # Training loop
    for epoch in range(latest_epoch, config.training.num_epoch):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            feat_mlp, logits, _ = model(images)
            if config.training.architecture_type == 'vit_pretrained':
                preds = logits.logits.argmax(dim=-1)
                loss = criterion(logits.logits, labels)
            else:
                _, preds = torch.max(logits, 1)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)

            # # for debug
            # from tools.image import save_img
            # save_img(images, batch_idx, epoch, training_path/"augmented")

        average_loss = running_loss / len(train_loader)
        average_acc = running_corrects.double() / len(train_loader.dataset)
        loss_values.append(average_loss)
        console.info(f"Epoch [{epoch + 1}/{config.training.num_epoch}] - Loss: {average_loss:.4f} "
                     f"- Training Acc: {average_acc:.4f}")
        plot_loss(loss_values, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path)

        # Update the learning rate
        if (epoch - latest_epoch) > 50 and loss_values[-1] > loss_values[-2]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        # save intermediate weight
        if (epoch + 1) % config.training.save_model_every_n_epoch == 0:
            # Save the model weights
            saved_weights = f'model_weights_epoch_{epoch + 1}.pth'
            saved_weights_file = training_path / saved_weights

            console.info(f"Model weights saved to {saved_weights_file}")
            torch.save(model.state_dict(), saved_weights_file)

    # Create a plot of the loss values
    plot_loss(loss_values, num_epoch=(config.training.num_epoch - latest_epoch), training_path=config.training_path)

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
                                  phase='test',
                                  gray=config.training.gray)

        val_loader = DataLoader(test_dataset,
                                batch_size=config.classifier.batch_size,
                                shuffle=True,
                                num_workers=4)
    else:
        console.quit('no data for testing model')

    # Evaluation loop
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels, img_names in val_loader:
            images, labels = images.to(device), labels.to(device)

            _, logits, _ = model(images)

            if config.training.architecture_type == 'vit_pretrained':
                preds = logits.logits.argmax(dim=-1)
            else:
                _, preds = torch.max(logits, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            save_image = False
            if save_image:
                for i in range(len(preds)):
                    int_label = preds[i].item()
                    string_label = val_loader.dataset.get_string_label(int_label)
                    image_name = img_names[i]
                    image_path = os.path.join(training_path, 'output/', string_label, image_name.replace('output/', ''))

                    if not os.path.exists(os.path.dirname(image_path)):
                        os.makedirs(os.path.dirname(image_path))

                    input_path = os.path.join(val_loader.dataset.root_dir, image_name)
                    shutil.copy(input_path, image_path)

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





