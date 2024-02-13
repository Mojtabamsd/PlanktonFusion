import datetime
from configs.config import Configuration
from tools.console import Console
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset.uvp_dataset import UvpDataset
from models.classifier_cnn import count_parameters
from models.autoencoder import ConvAutoencoder, ResNetCustom, ResNetAutoencoder
import torch
import pandas as pd
from tools.utils import report_to_df, memory_usage
import os
import shutil
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import xgboost as xgb
from feature_extraction.feature_uvpec import feature_uvpec
from tools.utils import plot_results


class ToTensorNoNormalize(object):
    def __call__(self, pic):
        img = torch.tensor(list(pic.tobytes()), dtype=torch.uint8)
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        img = img.permute(2, 0, 1).contiguous()
        return img


def classifier(config_path, input_path, output_path):

    config = Configuration(config_path, input_path, output_path)

    # Create output directory
    input_folder = Path(input_path)
    output_folder = Path(output_path)

    input_folder_train = input_folder / "train"
    input_folder_test = input_folder / "test"

    console = Console(output_folder)
    console.info("Classification started ...")

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
    rel_classification_path = Path("classification" + time_str)
    classification_path = output_folder / rel_classification_path
    if not classification_path.exists():
        classification_path.mkdir(exist_ok=True, parents=True)
    elif classification_path.exists():
        console.error("The output folder", classification_path, "exists.")
        console.quit("Folder exists, not overwriting previous results.")

    # Save configuration file
    output_config_filename = classification_path / "config.yaml"
    config.write(output_config_filename)

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((config.sampling.target_size[0], config.sampling.target_size[1])),  # Resize to desired input size
        transforms.ToTensor(),
    ])

    device = torch.device(f'cuda:{config.base.gpu_index}' if
                          torch.cuda.is_available() and config.base.cpu is False else 'cpu')
    console.info(f"Running on:  {device}")

    if config.classifier.feature_type == 'conv_autoencoder' or config.classifier.feature_type == 'resnet18':

        if config.classifier.feature_type == 'conv_autoencoder':
            model = ConvAutoencoder(latent_dim=config.autoencoder.latent_dim,
                                    input_size=config.sampling.target_size,
                                    gray=config.autoencoder.gray)
        elif config.classifier.feature_type == 'resnet18':
            model = ResNetCustom(num_classes=config.sampling.num_class,
                                 latent_dim=config.autoencoder.latent_dim,
                                 gray=config.autoencoder.gray)
        elif config.classifier.feature_type == 'resnet18_autoencoder':
            model = ResNetAutoencoder(latent_dim=config.autoencoder.latent_dim,
                                      input_size=config.sampling.target_size,
                                      gray=config.autoencoder.gray)

        transform = transforms.Compose([
            transforms.Resize((config.sampling.target_size[0], config.sampling.target_size[1])),
            # Resize to desired input size
            transforms.ToTensor(),
        ])

        # Calculate the number of parameters in millions
        num_params = count_parameters(model) / 1_000_000
        console.info(f"The model has approximately {num_params:.2f} million parameters.")

        model.to(device)

        # test memory usage
        console.info(memory_usage(config, model, device))

        # Save the model's state dictionary to a file
        saved_weights = "model_weights_final.pth"
        training_path = Path(config.classifier.path_model)
        saved_weights_file = training_path / saved_weights

        console.info("Deep learning latent features will be extracted")
        console.info("Model loaded from ", saved_weights_file)
        model.load_state_dict(torch.load(saved_weights_file, map_location=device))
        model.to(device)

    elif config.classifier.feature_type == 'uvpec':
        model = 'uvpec'
        console.info("Simple uvpec features will be extracted ")
        transform = transforms.Compose([
            transforms.Resize((config.sampling.target_size[0], config.sampling.target_size[1])),
            # Resize to desired input size
            ToTensorNoNormalize(),
        ])

    else:
        console.quit("Please select correct parameter for feature_type")

    train_dataset = UvpDataset(root_dir=input_folder_train,
                               num_class=config.sampling.num_class,
                               # csv_file=None,
                               csv_file=input_csv_train,
                               transform=transform,
                               phase='test')

    dataloader_train = DataLoader(train_dataset, batch_size=config.classifier.batch_size, shuffle=False)

    test_dataset = UvpDataset(root_dir=input_folder_test,
                              num_class=config.sampling.num_class,
                              # csv_file=None,
                              csv_file=input_csv_test,
                              transform=transform,
                              phase='test')

    dataloader_test = DataLoader(test_dataset, batch_size=config.classifier.batch_size, shuffle=False)

    sub_folder = input_folder_train / "output"
    # train_test_classifier(model, dataloader_train, classification_path, config, device, console, sub_folder)
    classifier_model = train_classifier(model, dataloader_train, config, device, console)
    test_classifier(model, classifier_model, dataloader_test, classification_path, config, device, console, sub_folder)


def train_test_classifier(model, dataloader, prediction_path, config, device, console, sub_folder):

    all_labels = []
    latent_vectors = []

    if model == 'uvpec':
        for index, (images, labels, img_names) in enumerate(dataloader):
            images_np = images.numpy()
            for img, img_name in zip(images_np, img_names):
                feature = feature_uvpec(np.squeeze(img), img_names)
                feature_values = np.array([feature_value for feature_value in feature.values()])
                latent_vectors.append(feature_values)
            all_labels.append(labels.data.cpu().detach().numpy())

        all_labels = np.concatenate(all_labels).ravel()
    else:
        model.eval()
        with torch.no_grad():
            for index, (images, labels, img_names) in enumerate(dataloader):
                images = images.to(device)
                _, latent = model(images)

                latent_vectors.extend(latent.cpu().numpy())
                all_labels.append(labels.data.cpu().detach().numpy())

            all_labels = np.concatenate(all_labels).ravel()

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(latent_vectors,
                                                        all_labels, test_size=0.2, random_state=42)

    if config.classifier.classifier_type == 'svm':
        # Train SVM classifier
        svm_classifier = SVC(kernel='rbf', class_weight='balanced')
        svm_classifier.fit(x_train, y_train)

        # Evaluate the SVM classifier
        y_pred = svm_classifier.predict(x_test)

    elif config.classifier.classifier_type == 'xgboost':
        xg_classifier = xgb.XGBClassifier()
        xg_classifier.fit(x_train, y_train)

        # Evaluate the Xgboost classifier
        y_pred = xg_classifier.predict(x_test)
    else:
        console.quit("Please select correct parameter for classifier_type")

    cl_report = classification_report(
        y_true=y_test,
        y_pred=y_pred,
        target_names=dataloader.dataset.label_to_int,
        digits=6,
    )

    conf_mtx = confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
    )

    cl_report_df = report_to_df(cl_report)
    report_filename = os.path.join(prediction_path, 'report.csv')
    cl_report_df.to_csv(report_filename)

    cm_df = pd.DataFrame(conf_mtx)
    conf_mtx_filename = os.path.join(prediction_path, 'conf_matrix.csv')
    cm_df.to_csv(conf_mtx_filename)

    plot_results(cl_report_df, conf_mtx, prediction_path, target_names=dataloader.dataset.label_to_int)

    # save feature.feather and dictionary idx if you want use for uvpec classifier

    df = pd.DataFrame(latent_vectors, columns=['latent{}'.format(i) for i in range(1, latent_vectors[0].size + 1)])
    int_to_label = {v: k for k, v in dataloader.dataset.label_to_int.items()}
    df['labels'] = [int_to_label[label] for label in all_labels]

    report_filename = os.path.join(prediction_path, 'features.feather')
    df.to_feather(report_filename)

    # split taxon names and their IDs from EcoTaxa

    _, folder_list, _ = next(os.walk(sub_folder))
    folder_list_with_underscore = [label + '_' for label in folder_list]
    folder_list_split = [folder.split('_') for folder in folder_list_with_underscore]
    dico_id = {folder_list_split[i][0]: folder_list_split[i][1] for i in range(len(folder_list_split))}
    dic_path = os.path.join(prediction_path, 'dico_id.npy')
    np.save(dic_path, dico_id)


def train_classifier(model, dataloader, config, device, console):

    all_labels = []
    latent_vectors = []

    if model == 'uvpec':
        for index, (images, labels, img_names) in enumerate(dataloader):
            images_np = images.numpy()
            for img, img_name in zip(images_np, img_names):
                feature = feature_uvpec(np.squeeze(img), img_names)
                feature_values = np.array([feature_value for feature_value in feature.values()])
                latent_vectors.append(feature_values)
            all_labels.append(labels.data.cpu().detach().numpy())

        all_labels = np.concatenate(all_labels).ravel()
    else:
        model.eval()
        with torch.no_grad():
            for index, (images, labels, img_names) in enumerate(dataloader):
                images = images.to(device)
                _, latent = model(images)

                latent_vectors.extend(latent.cpu().numpy())
                all_labels.append(labels.data.cpu().detach().numpy())

            all_labels = np.concatenate(all_labels).ravel()

    # # in case of using pre-generated latents
    # latent_vectors = np.load(config.input_path + r'\train\latents.npy')

    # Split data into training and testing sets
    x_train, y_train = latent_vectors, all_labels

    if config.classifier.classifier_type == 'svm':
        # Train SVM classifier
        svm_classifier = SVC(kernel='rbf', class_weight='balanced')
        svm_classifier.fit(x_train, y_train)

        return svm_classifier

    elif config.classifier.classifier_type == 'xgboost':

        xg_classifier = xgb.XGBClassifier()
        xg_classifier.fit(x_train, y_train)

        # # weighted sampling
        # class_weights = len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
        # sample_weights = np.array([class_weights[label] for label in y_train])
        # xg_classifier = xgb.XGBClassifier()
        # xg_classifier.fit(x_train, y_train, sample_weight=sample_weights)

        # # LOV setting
        # num_class = 13
        # random_state = 42
        # n_jobs = 12
        # learning_rate = 0.2
        # max_depth = 5
        # num_trees_CV = 500
        # dtrain = xgb.DMatrix(pd.DataFrame(x_train), label=y_train)
        #
        # xgb_params = {'nthread': n_jobs, 'eta': learning_rate, 'max_depth': max_depth, 'subsample': 0.75,
        #               'tree_method': 'hist', 'objective': 'multi:softprob',
        #               'eval_metric': ['mlogloss', 'merror'], 'num_class': num_class,
        #               'seed': random_state}
        #
        # xg_classifier = xgb.train(xgb_params, dtrain, num_boost_round=num_trees_CV)

        return xg_classifier

    elif config.classifier.classifier_type == 'isf':
        from sklearn.ensemble import IsolationForest

        clf = IsolationForest(random_state=42)
        normal_indices = [i for i, label in enumerate(y_train) if label != 12]
        x_normal = [x_train[i] for i in normal_indices]
        y_normal = np.array([y_train[i] for i in range(len(y_train)) if i in normal_indices])

        clf.fit(x_normal)

        xg_classifier = xgb.XGBClassifier()
        xg_classifier.fit(x_normal, y_normal)

        return xg_classifier, clf
    else:
        console.quit("Please select correct parameter for classifier_type")


def test_classifier(model, classifier_model, dataloader, prediction_path, config, device, console, sub_folder):

    all_labels = []
    latent_vectors = []

    if model == 'uvpec':
        for index, (images, labels, img_names) in enumerate(dataloader):
            images_np = images.numpy()
            for img, img_name in zip(images_np, img_names):
                feature = feature_uvpec(np.squeeze(img), img_names)
                feature_values = np.array([feature_value for feature_value in feature.values()])
                latent_vectors.append(feature_values)
            all_labels.append(labels.data.cpu().detach().numpy())

        all_labels = np.concatenate(all_labels).ravel()
    else:
        model.eval()
        with torch.no_grad():
            for index, (images, labels, img_names) in enumerate(dataloader):
                images = images.to(device)
                _, latent = model(images)

                latent_vectors.extend(latent.cpu().numpy())
                all_labels.append(labels.data.cpu().detach().numpy())

            all_labels = np.concatenate(all_labels).ravel()

    # # in case of using pre-generated latents
    # latent_vectors = np.load(config.input_path + r'\test\latents.npy')

    # testing sets
    x_test, y_test = latent_vectors, all_labels

    if config.classifier.classifier_type == 'svm':

        # Evaluate the SVM classifier
        y_pred = classifier_model.predict(x_test)

    elif config.classifier.classifier_type == 'xgboost':

        # Evaluate the Xgboost classifier
        y_pred = classifier_model.predict(x_test)

        # dtest = xgb.DMatrix(pd.DataFrame(x_test))
        # y_pred = classifier_model.predict(dtest)
        # y_pred = np.argmax(y_pred, axis=1)

    elif config.classifier.classifier_type == 'isf':
        xgb_classifier, clf = classifier_model
        y_pred_iso = clf.predict(x_test)

        y_pred_binary = [1 if label == 1 else -1 for label in y_pred_iso]
        detritus_indices = [i for i, label in enumerate(y_pred_binary) if label == -1]

        # Separate potential normal samples
        x_normal = [x_test[i] for i in range(len(y_test)) if i not in detritus_indices]

        y_normal_pred = xgb_classifier.predict(x_normal)

        y_pred = (np.ones(y_test.size) * 12).astype(int)
        c = 0
        for i in range(len(y_test)):
            if i not in detritus_indices:
                y_pred[i] = y_normal_pred[c]
                c += 1

    else:
        console.quit("Please select correct parameter for classifier_type")

    cl_report = classification_report(
        y_true=y_test,
        y_pred=y_pred,
        target_names=dataloader.dataset.label_to_int,
        digits=6,
    )

    conf_mtx = confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
    )

    cl_report_df = report_to_df(cl_report)
    report_filename = os.path.join(prediction_path, 'report.csv')
    cl_report_df.to_csv(report_filename)

    cm_df = pd.DataFrame(conf_mtx)
    conf_mtx_filename = os.path.join(prediction_path, 'conf_matrix.csv')
    cm_df.to_csv(conf_mtx_filename)

    plot_results(cl_report_df, conf_mtx, prediction_path, target_names=dataloader.dataset.label_to_int)

    # save feature.feather and dictionary idx if you want use for uvpec classifier

    df = pd.DataFrame(latent_vectors, columns=['latent{}'.format(i) for i in range(1, latent_vectors[0].size + 1)])
    int_to_label = {v: k for k, v in dataloader.dataset.label_to_int.items()}
    df['labels'] = [int_to_label[label] for label in all_labels]

    report_filename = os.path.join(prediction_path, 'features.feather')
    df.to_feather(report_filename)

    # split taxon names and their IDs from EcoTaxa

    _, folder_list, _ = next(os.walk(sub_folder))
    folder_list_with_underscore = [label + '_' for label in folder_list]
    folder_list_split = [folder.split('_') for folder in folder_list_with_underscore]
    dico_id = {folder_list_split[i][0]: folder_list_split[i][1] for i in range(len(folder_list_split))}
    dic_path = os.path.join(prediction_path, 'dico_id.npy')
    np.save(dic_path, dico_id)





