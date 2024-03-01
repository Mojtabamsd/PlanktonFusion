import yaml
from pathlib import Path


class BaseConfig:
    def __init__(self, cpu=False, all_gpu=False, gpu_index=0):
        self.cpu = cpu
        self.all_gpu = all_gpu
        self.gpu_index = gpu_index


class SamplingConfig:
    def __init__(self, path_uvp5, path_uvp6, path_uvp6_csv, path_output, uvp_type, num_class,
                 sampling_method, sampling_percent_uvp5, sampling_percent_uvp6, target_size,
                 test_dataset_sampling, test_percent_uvp6, test_percent_uvp5, create_folder):
        self.path_uvp5 = path_uvp5
        self.path_uvp6 = path_uvp6
        self.path_uvp6_csv = path_uvp6_csv
        self.path_output = path_output
        self.uvp_type = uvp_type
        self.num_class = num_class
        self.sampling_method = sampling_method
        self.sampling_percent_uvp5 = sampling_percent_uvp5
        self.sampling_percent_uvp6 = sampling_percent_uvp6
        self.target_size = target_size
        self.test_dataset_sampling = test_dataset_sampling
        self.test_percent_uvp6 = test_percent_uvp6
        self.test_percent_uvp5 = test_percent_uvp5
        self.create_folder = create_folder


class SamplingSynConfig:
    def __init__(self, path_syn, uvp_type, path_uvp6, path_uvp6_csv, path_output, labels_included, sampling_method,
                 sampling_percent_uvp6, target_size, test_dataset_sampling, test_percent_uvp6, create_folder):
        self.path_syn = path_syn
        self.uvp_type = uvp_type
        self.path_uvp6 = path_uvp6
        self.path_uvp6_csv = path_uvp6_csv
        self.labels_included = labels_included
        self.path_output = path_output
        self.sampling_method = sampling_method
        self.sampling_percent_uvp6 = sampling_percent_uvp6
        self.target_size = target_size
        self.test_dataset_sampling = test_dataset_sampling
        self.test_percent_uvp6 = test_percent_uvp6
        self.create_folder = create_folder


class TrainingConfig:
    def __init__(self, architecture_type, batch_size, gray, pre_train, learning_rate, num_epoch,
                 save_model_every_n_epoch, loss, path_pretrain):
        self.architecture_type = architecture_type
        self.batch_size = batch_size
        self.gray = gray
        self.pre_train = pre_train
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.save_model_every_n_epoch = save_model_every_n_epoch
        self.loss = loss
        self.path_pretrain = path_pretrain


class PredictionConfig:
    def __init__(self, path_model, batch_size):
        self.path_model = path_model
        self.batch_size = batch_size


class PredictionAutoConfig:
    def __init__(self, path_model_auto, path_model_class, architecture_type1, architecture_type2,
                 latent_dim, batch_size):
        self.path_model_auto = path_model_auto
        self.path_model_class = path_model_class
        self.architecture_type1 = architecture_type1
        self.architecture_type2 = architecture_type2
        self.latent_dim = latent_dim
        self.batch_size = batch_size


class AutoencoderConfig:
    def __init__(self, architecture_type, latent_dim, batch_size, gray, learning_rate, num_epoch,
                 save_model_every_n_epoch, loss, path_pretrain):
        self.architecture_type = architecture_type
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.gray = gray
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.save_model_every_n_epoch = save_model_every_n_epoch
        self.loss = loss
        self.path_pretrain = path_pretrain


class SSLConfig:
    def __init__(self, architecture_type, temperature, latent_dim, batch_size, gray, learning_rate, num_epoch,
                 save_model_every_n_epoch, loss):
        self.architecture_type = architecture_type
        self.temperature = temperature
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.gray = gray
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.save_model_every_n_epoch = save_model_every_n_epoch
        self.loss = loss


class ClassifierConfig:
    def __init__(self, path_model, batch_size, feature_type, classifier_type):
        self.path_model = path_model
        self.batch_size = batch_size
        self.feature_type = feature_type
        self.classifier_type = classifier_type


class MemoryConfig:
    def __init__(self, visual_embedded_model, loss, k, batch_size, num_epoch):
        self.visual_embedded_model = visual_embedded_model
        self.loss = loss
        self.k = k
        self.batch_size = batch_size
        self.num_epoch = num_epoch


class Configuration:
    def __init__(self, config_file_path, input_path=None, output_path=None):
        with open(config_file_path, "r") as config_file:
            config_data = yaml.safe_load(config_file)

        self.input_path = input_path
        self.output_path = output_path
        self.base = BaseConfig(**config_data['base'])
        self.sampling = SamplingConfig(**config_data['sampling'])
        self.sampling_syn = SamplingSynConfig(**config_data['sampling_syn'])
        self.training = TrainingConfig(**config_data['training'])
        self.prediction = PredictionConfig(**config_data['prediction'])
        self.prediction_auto = PredictionAutoConfig(**config_data['prediction_auto'])
        self.autoencoder = AutoencoderConfig(**config_data['autoencoder'])
        self.ssl = SSLConfig(**config_data['ssl'])
        self.classifier = ClassifierConfig(**config_data['classifier'])
        self.memory = MemoryConfig(**config_data['memory'])

    def write(self, filename):
        filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        with filename.open("w") as file_handler:
            yaml.dump(
                self, file_handler, allow_unicode=True, default_flow_style=False
            )

