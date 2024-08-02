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
from tools.utils import report_to_df, plot_loss, memory_usage
from tools.randaugment import rand_augment_transform
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomAffine, RandomResizedCrop, \
    ColorJitter, RandomGrayscale, RandomPerspective, RandomVerticalFlip
from tools.augmentation import GaussianNoise
from models.loss import FocalLoss, WeightedCrossEntropyLoss, LogitAdjustmentLoss, LogitAdjust
from transformers import ViTForImageClassification
from models.proco import ProCoLoss
import time


def train_contrastive(config_path, input_path, output_path):

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

    if config.training_contrastive.path_pretrain:
        training_path = Path(config.training_contrastive.path_pretrain)
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
    randaug_m = 10
    randaug_n = 2
    # ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    grayscale_mean = 128
    ra_params = dict(
        translate_const=int(config.sampling.target_size[0] * 0.45),
        img_mean=grayscale_mean
    )

    transform_base = [
        transforms.Resize((config.sampling.target_size[0], config.sampling.target_size[1])),
        RandomResizedCrop((config.sampling.target_size[0], config.sampling.target_size[1])),
        RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params, use_cmc=True),
        transforms.ToTensor(),
    ]
    transform_sim = [
        transforms.RandomResizedCrop(config.sampling.target_size[0]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ]

    transform_train = [transforms.Compose(transform_base), transforms.Compose(transform_sim),
                       transforms.Compose(transform_sim), ]

    transform_val = transforms.Compose([
        transforms.Resize((config.sampling.target_size[0], config.sampling.target_size[1])),
        transforms.ToTensor()
        ])

    # Create uvp dataset datasets for training and validation
    train_dataset = UvpDataset(root_dir=input_folder_train,
                               num_class=config.sampling.num_class,
                               csv_file=input_csv_train,
                               transform=transform_train,
                               phase=phase,
                               gray=config.training_contrastive.gray)

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
                                  batch_size=config.training_contrastive.batch_size,
                                  sampler=sampler)

    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=config.training_contrastive.batch_size,
                                  shuffle=True,
                                  num_workers=4)

    device = torch.device(f'cuda:{config.base.gpu_index}' if
                          torch.cuda.is_available() and config.base.cpu is False else 'cpu')
    console.info(f"Running on:  {device}")

    if config.training_contrastive.architecture_type == 'resnet50':
        model = resnext.Model(name='resnet50', num_classes=config.sampling.num_class,
                              feat_dim=config.training_contrastive.feat_dim,
                              use_norm=config.training_contrastive.use_norm,
                              gray=config.training_contrastive.gray)

    elif config.training_contrastive.architecture_type == 'resnext50':
        model = resnext.Model(name='resnext50', num_classes=config.sampling.num_class,
                              feat_dim=config.training_contrastive.feat_dim,
                              use_norm=config.training_contrastive.use_norm,
                              gray=config.training_contrastive.gray)

    else:
        console.quit("Please select correct parameter for architecture_type")

    # Calculate the number of parameters in millions
    num_params = count_parameters(model) / 1_000_000
    console.info(f"The model has approximately {num_params:.2f} million parameters.")

    model.to(device)

    # test memory usage
    # console.info(memory_usage(config, model, device))

    if config.training_contrastive.path_pretrain:
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
    if config.training_contrastive.loss == 'proco':
        criterion_ce = LogitAdjust(class_counts).to(device)
        criterion_scl = ProCoLoss(contrast_dim=config.training_contrastive.feat_dim,
                                  temperature=config.training_contrastive.temp,
                                  num_classes=config.sampling.num_class).to(device)

    optimizer = torch.optim.SGD(model.parameters(), config.training_contrastive.learning_rate,
                                momentum=config.training_contrastive.momentum,
                                weight_decay=config.training_contrastive.weight_decay)

    if config.training_contrastive.num_epoch == 200:
        config.training_contrastive.schedule = [160, 180]
        config.training_contrastive.warmup_epochs = 5
    elif config.training_contrastive.num_epoch == 400:
        config.training_contrastive.schedule = [360, 380]
        config.training_contrastive.warmup_epochs = 10
    else:
        config.training_contrastive.schedule = [config.training_contrastive.num_epoch * 0.8, config.training_contrastive.num_epoch * 0.9]
        config.training_contrastive.warmup_epochs = 5 * config.training_contrastive.num_epoch // 200

    loss_values = []

    # Training loop
    for epoch in range(latest_epoch, config.training_contrastive.num_epoch):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        batch_time = AverageMeter('Time', ':6.3f')
        ce_loss_all = AverageMeter('CE_Loss', ':.4e')
        scl_loss_all = AverageMeter('SCL_Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        end = time.time()


        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = torch.cat([images[0], images[1], images[2]], dim=0)
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.shape[0]

            feat_mlp, ce_logits, _ = model(images)
            _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
            ce_logits, _, __ = torch.split(ce_logits, [batch_size, batch_size, batch_size], dim=0)

            contrast_logits1 = criterion_scl(f2, labels)
            contrast_logits2 = criterion_scl(f3, labels)
            contrast_logits = (contrast_logits1 + contrast_logits2) / 2

            scl_loss = (criterion_ce(contrast_logits1, labels) + criterion_ce(contrast_logits2, labels)) / 2
            ce_loss = criterion_ce(ce_logits, labels)

            alpha = 1
            logits = ce_logits + alpha * contrast_logits
            loss = ce_loss + alpha * scl_loss

            ce_loss_all.update(ce_loss.item(), batch_size)
            scl_loss_all.update(scl_loss.item(), batch_size)

            acc1 = accuracy(logits, labels, topk=(1,))
            top1.update(acc1[0].item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # # for debug
            # from tools.image import save_img
            # save_img(images, batch_idx, epoch, training_path/"augmented")

            if batch_idx % 20 == 0:
                output = ('Epoch: [{0}][{1}/{2}] \t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                          'SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, batch_idx, len(train_loader), batch_time=batch_time,
                    ce_loss=ce_loss_all, scl_loss=scl_loss_all, top1=top1, ))  # TODO
                print(output)

        console.info(f"CE loss train [{epoch + 1}/{config.training_contrastive.num_epoch}] - Loss: {ce_loss_all.avg:.4f} ")
        console.info(f"SCL loss train [{epoch + 1}/{config.training_contrastive.num_epoch}] - Loss: {scl_loss_all.avg:.4f} ")
        console.info(f"acc train top1 [{epoch + 1}/{config.training_contrastive.num_epoch}] - Acc: {top1.avg:.4f} ")

        plot_loss(ce_loss_all, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path, name='CE_loss.png')
        plot_loss(scl_loss_all, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path, name='SCL_loss.png')
        plot_loss(top1, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path, name='ACC.png')

        # Update the learning rate
        if (epoch - latest_epoch) > 50 and loss_values[-1] > loss_values[-2]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

        # save intermediate weight
        if (epoch + 1) % config.training_contrastive.save_model_every_n_epoch == 0:
            # Save the model weights
            saved_weights = f'model_weights_epoch_{epoch + 1}.pth'
            saved_weights_file = training_path / saved_weights

            console.info(f"Model weights saved to {saved_weights_file}")
            torch.save(model.state_dict(), saved_weights_file)

    # Create a plot of the loss values
    plot_loss(ce_loss_all, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='CE_loss.png')
    plot_loss(scl_loss_all, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='SCL_loss.png')
    plot_loss(top1, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='ACC.png')

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
                                batch_size=config.training_contrastive.batch_size,
                                shuffle=True)

    elif input_csv_test is not None:
        console.info('Testing model with folder test')

        test_dataset = UvpDataset(root_dir=input_folder_test,
                                  num_class=config.sampling.num_class,
                                  csv_file=input_csv_test,
                                  transform=transform_val,
                                  phase='test',
                                  gray=config.training_contrastive.gray)

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

            outputs = model(images)

            if config.training_contrastive.architecture_type == 'vit_pretrained':
                preds = outputs.logits.argmax(dim=-1)
            else:
                _, preds = torch.max(outputs, 1)

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



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


