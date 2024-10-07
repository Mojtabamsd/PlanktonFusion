import datetime
from configs.config import Configuration
from tools.console import Console
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from dataset.uvp_dataset import UvpDataset
from models.classifier_cnn import count_parameters
from models import resnext
from dataset.imagenet import ImageNetLT
from dataset.inat import INaturalist
import math
import os
import shutil
import torch
import torch.distributed as dist
from tools.utils import report_to_df, plot_loss, shot_acc
from tools.randaugment import rand_augment_transform
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from tools.augmentation import ResizeAndPad
from models.loss import LogitAdjust
from models.proco import ProCoLoss
from models.procom import ProCoMLoss
import time
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def train_contrastive(config_path, input_path, output_path):

    config = Configuration(config_path, input_path, output_path)
    config.phase = 'train'      # will train with whole dataset and testing results if there is a test file
    # phase = 'train_val'  # will train with 80% dataset and testing results with the rest 20% of data

    config.input_path = input_path

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    if config.training_contrastive.dataset == 'uvp':
        input_folder_train = input_folder / "train"
        input_folder_test = input_folder / "test"
    elif config.training_contrastive.dataset == 'imagenet':
        input_folder_train = input_folder
        input_folder_test = input_folder

    console = Console(output_folder)
    console.info("Training started ...")

    sampled_images_csv_filename = "sampled_images.csv"
    input_csv_train = input_folder_train / sampled_images_csv_filename
    input_csv_test = input_folder_test / sampled_images_csv_filename

    config.input_folder_train = str(input_folder_train)
    config.input_folder_test = str(input_folder_test)
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
        rel_training_path = Path("training_contrastive" + time_str)
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

    config.training_path = str(training_path)

    # parallel processing
    # config.world_size = torch.cuda.device_count()

    if config.base.all_gpu:
        world_size = torch.cuda.device_count()
        console.info(f"Number of GPU available:  {world_size}")
    else:
        world_size = 1

    # dist.init_process_group(backend='gloo', init_method='env://', world_size=config.world_size, rank=rank)

    if config.training_contrastive.dataset == 'uvp':
        if world_size > 1:
            mp.spawn(train_uvp, args=(world_size, config, console), nprocs=world_size, join=True)
        else:
            train_uvp(config.base.gpu_index, world_size, config, console)

    elif config.training_contrastive.dataset == 'imagenet' or config.training_contrastive.dataset == 'inat':
        if world_size > 1:
            mp.spawn(train_imagenet_inatural, args=(world_size, config, console), nprocs=world_size, join=True)
        else:
            train_imagenet_inatural(config.base.gpu_index, world_size, config, console)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',  # Use 'gloo' or 'nccl' for multi-GPU
            init_method='env://',
            rank=rank,
            world_size=world_size
        )


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_uvp(rank, world_size, config, console):

    if world_size > 1:
        setup(rank, world_size)

    is_distributed = world_size > 1

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    console.info(f"Running on:  {device}")

    # Define data transformations
    randaug_m = 10
    randaug_n = 2
    # ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    grayscale_mean = 128
    ra_params = dict(
        translate_const=int(config.training_contrastive.target_size[0] * 0.45),
        img_mean=grayscale_mean
    )

    if config.training_contrastive.padding:
        resize_operation = ResizeAndPad((config.training_contrastive.target_size[0],
                                         config.training_contrastive.target_size[1]))
    else:
        resize_operation = transforms.Resize((config.training_contrastive.target_size[0],
                                              config.training_contrastive.target_size[1]))

    transform_base = [
        resize_operation,
        transforms.RandomResizedCrop(config.training_contrastive.target_size[0]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params, use_cmc=True),
        transforms.ToTensor(),
    ]
    transform_sim = [
        resize_operation,
        transforms.RandomResizedCrop(config.training_contrastive.target_size[0]),
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
        resize_operation,
        transforms.ToTensor()
        ])

    # Create uvp dataset datasets for training and validation
    train_dataset = UvpDataset(root_dir=config.input_folder_train,
                               num_class=config.sampling.num_class,
                               csv_file=config.input_csv_train,
                               transform=transform_train,
                               phase=config.phase,
                               gray=config.training_contrastive.gray)

    class_counts = train_dataset.data_frame['label'].value_counts().sort_index().tolist()
    total_samples = sum(class_counts)
    class_weights = [total_samples / (config.sampling.num_class * count) for count in class_counts]
    class_weights_tensor = torch.FloatTensor(class_weights)
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum()

    if is_distributed:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=config.training_contrastive.batch_size,
                              sampler=sampler,
                              shuffle=(not is_distributed),
                              num_workers=config.training_contrastive.num_workers)

    model = resnext.Model(name=config.training_contrastive.architecture_type, num_classes=config.sampling.num_class,
                          feat_dim=config.training_contrastive.feat_dim,
                          use_norm=config.training_contrastive.use_norm,
                          gray=config.training_contrastive.gray)

    # Calculate the number of parameters in millions
    num_params = count_parameters(model) / 1_000_000
    console.info(f"The model has approximately {num_params:.2f} million parameters.")

    model.to(device)
    # test memory usage
    # console.info(memory_usage(config, model, device))

    if config.training_contrastive.path_pretrain:
        pth_files = [file for file in os.listdir(config.training_path) if
                     file.endswith('.pth') and file != 'model_weights_final.pth']
        epochs = [int(file.split('_')[-1].split('.')[0]) for file in pth_files]
        latest_epoch = max(epochs)
        latest_pth_file = f"model_weights_epoch_{latest_epoch}.pth"

        saved_weights_file = os.path.join(config.training_path, latest_pth_file)

        console.info("Model loaded from ", saved_weights_file)
        model.load_state_dict(torch.load(saved_weights_file, map_location=device))
        model.to(device)
    else:
        latest_epoch = 0

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Loss criterion and optimizer
    if config.training_contrastive.loss == 'proco':
        criterion_ce = LogitAdjust(class_counts, device=device)
        criterion_scl = ProCoLoss(contrast_dim=config.training_contrastive.feat_dim,
                                  temperature=config.training_contrastive.temp,
                                  num_classes=config.sampling.num_class,
                                  device=device)
    elif config.training_contrastive.loss == 'procom':
        criterion_ce = LogitAdjust(class_counts, device=device)
        criterion_scl = ProCoMLoss(contrast_dim=config.training_contrastive.feat_dim,
                                   temperature=config.training_contrastive.temp,
                                   num_classes=config.sampling.num_class,
                                   max_modes=config.training_contrastive.max_modes,
                                   device=device)

    optimizer = torch.optim.SGD(model.parameters(), config.training_contrastive.learning_rate,
                                momentum=config.training_contrastive.momentum,
                                weight_decay=config.training_contrastive.weight_decay)

    # if config.training_contrastive.num_epoch == 200:
    #     config.training_contrastive.schedule = [160, 180]
    #     config.training_contrastive.warmup_epochs = 5
    # elif config.training_contrastive.num_epoch == 400:
    #     config.training_contrastive.schedule = [360, 380]
    #     config.training_contrastive.warmup_epochs = 10
    # else:
    #     config.training_contrastive.schedule = [config.training_contrastive.num_epoch * 0.8, config.training_contrastive.num_epoch * 0.9]
    #     config.training_contrastive.warmup_epochs = 5 * config.training_contrastive.num_epoch // 200

    ce_loss_all_avg = []
    scl_loss_all_avg = []
    top1_avg = []

    # Training loop
    for epoch in range(latest_epoch, config.training_contrastive.num_epoch):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        if is_distributed and sampler is not None:
            sampler.set_epoch(epoch)

        adjust_lr(optimizer, epoch, config)

        batch_time = AverageMeter('Time', ':6.3f')
        ce_loss_all = AverageMeter('CE_Loss', ':.4e')
        scl_loss_all = AverageMeter('SCL_Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        end = time.time()

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            batch_size = labels.shape[0]
            labels = labels.to(device)

            mini_batch_size = batch_size // config.training_contrastive.accumulation_steps

            images_0_mini_batches = torch.split(images[0], mini_batch_size)
            images_1_mini_batches = torch.split(images[1], mini_batch_size)
            images_2_mini_batches = torch.split(images[2], mini_batch_size)
            labels_mini_batches = torch.split(labels, mini_batch_size)

            optimizer.zero_grad()

            aggregated_logits = []

            for i in range(len(images_0_mini_batches)):
                mini_images = torch.cat([images_0_mini_batches[i], images_1_mini_batches[i], images_2_mini_batches[i]],
                                        dim=0)
                mini_labels = labels_mini_batches[i]
                mini_images, mini_labels = mini_images.to(device), mini_labels.to(device)

                feat_mlp, ce_logits, _ = model(mini_images)
                _, f2, f3 = torch.split(feat_mlp, [mini_batch_size, mini_batch_size, mini_batch_size], dim=0)
                ce_logits, _, __ = torch.split(ce_logits, [mini_batch_size, mini_batch_size, mini_batch_size], dim=0)

                contrast_logits1 = criterion_scl(f2, mini_labels)
                contrast_logits2 = criterion_scl(f3, mini_labels)
                contrast_logits1, contrast_logits2 = contrast_logits1.to(device), contrast_logits2.to(device)

                contrast_logits = (contrast_logits1 + contrast_logits2) / 2

                scl_loss = (criterion_ce(contrast_logits1, mini_labels) + criterion_ce(contrast_logits2, mini_labels)) / 2
                ce_loss = criterion_ce(ce_logits, mini_labels)

                alpha = 1
                logits = ce_logits + alpha * contrast_logits
                loss = ce_loss + alpha * scl_loss

                # Accumulate gradients
                loss.backward()
                aggregated_logits.append(logits)

            optimizer.step()
            aggregated_logits = torch.cat(aggregated_logits, dim=0)
            aggregated_logits = aggregated_logits.to(device)

            ce_loss_all.update(ce_loss.item(), batch_size)
            scl_loss_all.update(scl_loss.item(), batch_size)

            acc1 = accuracy(aggregated_logits, labels, topk=(1,))
            top1.update(acc1[0].item(), batch_size)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

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

        ce_loss_all_avg.append(ce_loss_all.avg)
        scl_loss_all_avg.append(scl_loss_all.avg)
        top1_avg.append(top1.avg)

        plot_loss(ce_loss_all_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path, name='CE_loss.png')
        plot_loss(scl_loss_all_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path, name='SCL_loss.png')
        plot_loss(top1_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path, name='ACC.png')

        # save intermediate weight
        if (epoch + 1) % config.training_contrastive.save_model_every_n_epoch == 0:
            # Save the model weights
            saved_weights = f'model_weights_epoch_{epoch + 1}.pth'
            saved_weights_file = os.path.join(config.training_path, saved_weights)

            console.info(f"Model weights saved to {saved_weights_file}")
            torch.save(model.state_dict(), saved_weights_file)

        if is_distributed:
            dist.barrier()

    if rank == 0:
        # Create a plot of the loss values
        plot_loss(ce_loss_all_avg, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='CE_loss.png')
        plot_loss(scl_loss_all_avg, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='SCL_loss.png')
        plot_loss(top1_avg, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='ACC.png')

        # Save the model's state dictionary to a file
        saved_weights = "model_weights_final.pth"
        saved_weights_file = os.path.join(config.training_path, saved_weights)

        torch.save(model.state_dict(), saved_weights_file)

        console.info(f"Final model weights saved to {saved_weights_file}")

    if is_distributed:
        dist.barrier()

    if rank == 0:
        # Create uvp dataset datasets for training and validation
        if config.phase == 'train_val':
            console.info('Testing model with validation subset')
            train_dataset.phase = 'val'
            val_dataset = train_dataset

            val_loader = DataLoader(val_dataset,
                                    batch_size=config.training_contrastive.batch_size,
                                    shuffle=True)

        elif config.input_csv_test is not None:
            console.info('Testing model with folder test')

            test_dataset = UvpDataset(root_dir=config.input_folder_test,
                                      num_class=config.sampling.num_class,
                                      csv_file=config.input_csv_test,
                                      transform=transform_val,
                                      phase='test',
                                      gray=config.training_contrastive.gray)

            val_loader = DataLoader(test_dataset,
                                    batch_size=config.classifier.batch_size,
                                    shuffle=True,
                                    num_workers=config.training_contrastive.num_workers)
        else:
            console.quit('no data for testing model')

        # Evaluation loop
        model.eval()
        all_labels = []
        all_preds = []
        batch_time = AverageMeter('Time', ':6.3f')
        ce_loss_all = AverageMeter('CE_Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        total_logits = torch.empty((0, config.sampling.num_class)).to(device)
        total_labels = torch.empty(0, dtype=torch.long).to(device)

        with torch.no_grad():
            end = time.time()
            for images, labels, img_names in val_loader:
                images, labels = images.to(device), labels.to(device)

                _, ce_logits, _ = model(images)
                logits = ce_logits

                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, labels))

                batch_time.update(time.time() - end)

                probs, preds = F.softmax(logits, dim=1).max(dim=1)
                save_image = False
                if save_image:
                    for i in range(len(preds)):
                        int_label = preds[i].item()
                        string_label = val_loader.dataset.get_string_label(int_label)
                        image_name = img_names[i]
                        image_path = os.path.join(config.training_path, 'output/', string_label, image_name.replace('output/', ''))

                        if not os.path.exists(os.path.dirname(image_path)):
                            os.makedirs(os.path.dirname(image_path))

                        input_path = os.path.join(val_loader.dataset.root_dir, image_name)
                        shutil.copy(input_path, image_path)

            # total_logits_list = [torch.zeros_like(total_logits) for _ in range(config.world_size)]
            # total_labels_list = [torch.zeros_like(total_labels) for _ in range(config.world_size)]
            #
            # dist.all_gather(total_logits_list, total_logits)
            # dist.all_gather(total_labels_list, total_labels)
            #
            # total_logits = torch.cat(total_logits_list, dim=0)
            # total_labels = torch.cat(total_labels_list, dim=0)

            ce_loss = criterion_ce(total_logits, total_labels)
            acc1 = accuracy(total_logits, total_labels, topk=(1,))

            ce_loss_all.update(ce_loss.item(), 1)
            top1.update(acc1[0].item(), 1)

            # if tf_writer is not None:
            #     tf_writer.add_scalar('CE loss/val', ce_loss_all.avg, epoch)
            #     tf_writer.add_scalar('acc/val_top1', top1.avg, epoch)

            all_probs, all_preds = F.softmax(total_logits, dim=1).max(dim=1)
            many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(all_preds, total_labels, train_loader, acc_per_cls=False)
            acc1 = top1.avg
            many = many_acc_top1*100
            med = median_acc_top1*100
            few = low_acc_top1*100
            # print('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(acc1, many, med, few))
            console.info('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(acc1, many, med, few))

        total_labels = total_labels.cpu().numpy()
        all_preds = all_preds.cpu().numpy()

        report = classification_report(
            total_labels,
            all_preds,
            target_names=train_dataset.label_to_int,
            digits=6,
        )

        conf_mtx = confusion_matrix(
            total_labels,
            all_preds,
        )

        df = report_to_df(report)
        report_filename = os.path.join(config.training_path, 'report_evaluation.csv')
        df.to_csv(report_filename)

        df = pd.DataFrame(conf_mtx)
        conf_mtx_filename = os.path.join(config.training_path, 'conf_matrix_evaluation.csv')
        df.to_csv(conf_mtx_filename)

        console.info('************* Evaluation Report *************')
        console.info(report)
        console.save_log(config.training_path)


def train_imagenet_inatural(rank, world_size, config, console):

    if world_size > 1:
        setup(rank, world_size)

    is_distributed = world_size > 1

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    console.info(f"Running on:  {device}")

    # number of classes for imagenet or inat
    if config.training_contrastive.dataset == 'inat':
        config.sampling.num_classes = 8142

        txt_train = f'iNaturalist18/iNaturalist18_train.txt'
        normalize = transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))

    elif config.training_contrastive.dataset == 'imagenet':
        config.sampling.num_classes = 1000

        txt_train = f'ImageNet_LT/ImageNet_LT_train.txt'
        # txt_val = f'ImageNet_LT/ImageNet_LT_val.txt'
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # Define data transformations
    randaug_m = 10
    randaug_n = 2
    cl_views = 'sim-sim'
    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(config.training_contrastive.target_size[0] * 0.45),
                     img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randncls = [
        transforms.RandomResizedCrop(config.training_contrastive.target_size[0], scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(config.training_contrastive.target_size[0]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_sim = [
        transforms.RandomResizedCrop(config.training_contrastive.target_size[0]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ]
    if cl_views == 'sim-sim':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim),
                           transforms.Compose(augmentation_sim), ]
    elif cl_views == 'sim-rand':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_sim), ]
    elif cl_views == 'rand-rand':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_randnclsstack), ]
    else:
        raise NotImplementedError("This augmentations strategy is not available for contrastive learning branch!")
    transform_val = transforms.Compose([
        transforms.Resize(config.training_contrastive.target_size[0]),
        transforms.CenterCrop(config.training_contrastive.target_size[0]),
        transforms.ToTensor(),
        normalize
    ])

    # config.input_path = ''
    # val_dataset = ImageNetLT(
    #     root=config.input_path,
    #     txt=txt_val,
    #     transform=transform_val, train=False)

    if config.training_contrastive.dataset == 'inat':
        train_dataset = INaturalist(
            root=config.input_folder_train,
            txt=txt_train,
            transform=transform_train)
    elif config.training_contrastive.dataset == 'imagenet':
        train_dataset = ImageNetLT(
            root=config.input_folder_train,
            txt=txt_train,
            transform=transform_train)

    console.info(f'===> Training data length {len(train_dataset)}')
    # console.info(f'===> Validation data length {len(val_dataset)}')

    if is_distributed:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=config.training_contrastive.batch_size,
                              sampler=sampler,
                              shuffle=(not is_distributed),
                              num_workers=config.training_contrastive.num_workers)

    # val_loader = DataLoader(
    #     val_dataset, batch_size=config.training_contrastive.batch_size, shuffle=False,
    #     num_workers=config.training_contrastive.num_workers, pin_memory=True, sampler=val_sampler)

    model = resnext.Model(name=config.training_contrastive.architecture_type, num_classes=config.sampling.num_classes,
                          feat_dim=config.training_contrastive.feat_dim,
                          use_norm=config.training_contrastive.use_norm,
                          gray=config.training_contrastive.gray)

    # Calculate the number of parameters in millions
    num_params = count_parameters(model) / 1_000_000
    console.info(f"The model has approximately {num_params:.2f} million parameters.")

    model.to(device)

    # test memory usage
    # console.info(memory_usage(config, model, device))

    if config.training_contrastive.path_pretrain:
        pth_files = [file for file in os.listdir(config.training_path) if
                     file.endswith('.pth') and file != 'model_weights_final.pth']
        epochs = [int(file.split('_')[-1].split('.')[0]) for file in pth_files]
        latest_epoch = max(epochs)
        latest_pth_file = f"model_weights_epoch_{latest_epoch}.pth"

        saved_weights_file = os.path.join(config.training_path, latest_pth_file)

        console.info("Model loaded from ", saved_weights_file)
        model.load_state_dict(torch.load(saved_weights_file, map_location=device))
        model.to(device)
    else:
        latest_epoch = 0

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Loss criterion and optimizer
    cls_num_list = train_dataset.cls_num_list
    config.cls_num = len(cls_num_list)

    if config.training_contrastive.loss == 'proco':
        criterion_ce = LogitAdjust(cls_num_list, device=device)
        criterion_scl = ProCoLoss(contrast_dim=config.training_contrastive.feat_dim,
                                  temperature=config.training_contrastive.temp,
                                  num_classes=config.sampling.num_classes,
                                  device=device)
    elif config.training_contrastive.loss == 'procom':
        criterion_ce = LogitAdjust(cls_num_list, device=device)
        criterion_scl = ProCoMLoss(contrast_dim=config.training_contrastive.feat_dim,
                                   temperature=config.training_contrastive.temp,
                                   num_classes=config.sampling.num_classes,
                                   max_modes=config.training_contrastive.max_modes,
                                   device=device)

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

    ce_loss_all_avg = []
    scl_loss_all_avg = []
    top1_avg = []

    # Training loop
    for epoch in range(latest_epoch, config.training_contrastive.num_epoch):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        if is_distributed and sampler is not None:
            sampler.set_epoch(epoch)

        adjust_lr(optimizer, epoch, config)

        batch_time = AverageMeter('Time', ':6.3f')
        ce_loss_all = AverageMeter('CE_Loss', ':.4e')
        scl_loss_all = AverageMeter('SCL_Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        end = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_size = labels.shape[0]
            labels = labels.to(device)

            mini_batch_size = batch_size // config.training_contrastive.accumulation_steps

            images_0_mini_batches = torch.split(images[0], mini_batch_size)
            images_1_mini_batches = torch.split(images[1], mini_batch_size)
            images_2_mini_batches = torch.split(images[2], mini_batch_size)
            labels_mini_batches = torch.split(labels, mini_batch_size)

            optimizer.zero_grad()

            aggregated_logits = []

            for i in range(len(images_0_mini_batches)):
                mini_images = torch.cat([images_0_mini_batches[i], images_1_mini_batches[i], images_2_mini_batches[i]],
                                        dim=0)
                mini_labels = labels_mini_batches[i]

                mini_images, mini_labels = mini_images.to(device), mini_labels.to(device)

                feat_mlp, ce_logits, _ = model(mini_images)
                _, f2, f3 = torch.split(feat_mlp, [mini_batch_size, mini_batch_size, mini_batch_size], dim=0)
                ce_logits, _, __ = torch.split(ce_logits, [mini_batch_size, mini_batch_size, mini_batch_size], dim=0)

                contrast_logits1 = criterion_scl(f2, mini_labels)
                contrast_logits2 = criterion_scl(f3, mini_labels)
                contrast_logits1, contrast_logits2 = contrast_logits1.to(device), contrast_logits2.to(device)

                contrast_logits = (contrast_logits1 + contrast_logits2) / 2

                scl_loss = (criterion_ce(contrast_logits1, mini_labels) + criterion_ce(contrast_logits2, mini_labels)) / 2
                ce_loss = criterion_ce(ce_logits, mini_labels)

                alpha = 1
                logits = ce_logits + alpha * contrast_logits
                loss = ce_loss + alpha * scl_loss

                # Accumulate gradients
                loss.backward()
                aggregated_logits.append(logits)

            optimizer.step()
            aggregated_logits = torch.cat(aggregated_logits, dim=0)
            aggregated_logits = aggregated_logits.to(device)

            ce_loss_all.update(ce_loss.item(), batch_size)
            scl_loss_all.update(scl_loss.item(), batch_size)

            acc1 = accuracy(aggregated_logits, labels, topk=(1,))
            top1.update(acc1[0].item(), batch_size)

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

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

        ce_loss_all_avg.append(ce_loss_all.avg)
        scl_loss_all_avg.append(scl_loss_all.avg)
        top1_avg.append(top1.avg)

        plot_loss(ce_loss_all_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path, name='CE_loss.png')
        plot_loss(scl_loss_all_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path, name='SCL_loss.png')
        plot_loss(top1_avg, num_epoch=(epoch - latest_epoch) + 1, training_path=config.training_path, name='ACC.png')

        # save intermediate weight
        if (epoch + 1) % config.training_contrastive.save_model_every_n_epoch == 0:
            # Save the model weights
            saved_weights = f'model_weights_epoch_{epoch + 1}.pth'
            saved_weights_file = os.path.join(config.training_path, saved_weights)

            console.info(f"Model weights saved to {saved_weights_file}")
            torch.save(model.state_dict(), saved_weights_file)

        if is_distributed:
            dist.barrier()

    if rank == 0:
        # Create a plot of the loss values
        plot_loss(ce_loss_all_avg, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='CE_loss.png')
        plot_loss(scl_loss_all_avg, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='SCL_loss.png')
        plot_loss(top1_avg, num_epoch=(config.training_contrastive.num_epoch - latest_epoch), training_path=config.training_path, name='ACC.png')

        # Save the model's state dictionary to a file
        saved_weights = "model_weights_final.pth"
        saved_weights_file = os.path.join(config.training_path, saved_weights)

        torch.save(model.state_dict(), saved_weights_file)

        console.info(f"Final model weights saved to {saved_weights_file}")

    if is_distributed:
        dist.barrier()

    if rank == 0:
        if config.training_contrastive.dataset == 'inat':
            txt_test = f'iNaturalist18/iNaturalist18_val.txt'
            test_dataset = INaturalist(
                root=config.input_path,
                txt=txt_test,
                transform=transform_val, train=False)
        elif config.training_contrastive.dataset == 'imagenet':
            txt_test = f'ImageNet_LT/ImageNet_LT_test.txt'
            test_dataset = ImageNetLT(
                root=config.input_path,
                txt=txt_test,
                transform=transform_val, train=False)

        test_loader = DataLoader(test_dataset,
                                 batch_size=config.training_contrastive.batch_size,
                                 shuffle=True,
                                 num_workers=config.training_contrastive.num_workers)

        # Evaluation loop
        model.eval()
        all_labels = []
        all_preds = []
        batch_time = AverageMeter('Time', ':6.3f')
        ce_loss_all = AverageMeter('CE_Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        total_logits = torch.empty((0, config.sampling.num_classes)).to(device)
        total_labels = torch.empty(0, dtype=torch.long).to(device)

        with torch.no_grad():
            end = time.time()
            for i, data in enumerate(test_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                _, ce_logits, _ = model(images)
                logits = ce_logits

                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, labels))

                batch_time.update(time.time() - end)

            ce_loss = criterion_ce(total_logits, total_labels)
            acc1 = accuracy(total_logits, total_labels, topk=(1,))

            ce_loss_all.update(ce_loss.item(), 1)
            top1.update(acc1[0].item(), 1)

            # if tf_writer is not None:
            #     tf_writer.add_scalar('CE loss/val', ce_loss_all.avg, epoch)
            #     tf_writer.add_scalar('acc/val_top1', top1.avg, epoch)

            all_probs, all_preds = F.softmax(total_logits, dim=1).max(dim=1)
            many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(all_preds, total_labels, train_loader, acc_per_cls=False)
            acc1 = top1.avg
            many = many_acc_top1*100
            med = median_acc_top1*100
            few = low_acc_top1*100
            # print('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(acc1, many, med, few))
            console.info('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(acc1, many, med, few))

        total_labels = total_labels.cpu().numpy()
        all_preds = all_preds.cpu().numpy()

        report = classification_report(
            total_labels,
            all_preds,
            digits=6,
        )

        conf_mtx = confusion_matrix(
            total_labels,
            all_preds,
        )

        df = report_to_df(report)
        report_filename = os.path.join(config.training_path, 'report_evaluation.csv')
        df.to_csv(report_filename)

        df = pd.DataFrame(conf_mtx)
        conf_mtx_filename = os.path.join(config.training_path, 'conf_matrix_evaluation.csv')
        df.to_csv(conf_mtx_filename)

        console.info('************* Evaluation Report *************')
        console.info(report)
        console.save_log(config.training_path)


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


def adjust_lr(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config.training_contrastive.learning_rate
    cos = False
    if epoch < config.training_contrastive.warmup_epochs:
        lr = lr / config.training_contrastive.warmup_epochs  * (epoch + 1)
    elif cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - config.training_contrastive.warmup_epochs + 1) /
                                   (config.training_contrastive.num_epoch - config.training_contrastive.warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in config.training_contrastive.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


