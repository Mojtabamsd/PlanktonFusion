import yaml
from pathlib import Path


class BaseConfig:
    def __init__(self, cpu=False, gpu_index=0):
        self.cpu = cpu
        self.gpu_index = gpu_index


class SamplingConfig:
    def __init__(self, path_uvp5, path_uvp6, path_output, uvp_type, num_class,
                 sampling_method, sampling_percent_uvp5, sampling_percent_uvp6, target_size,
                 test_dataset_sampling, test_percent):
        self.path_uvp5 = path_uvp5
        self.path_uvp6 = path_uvp6
        self.path_output = path_output
        self.uvp_type = uvp_type
        self.num_class = num_class
        self.sampling_method = sampling_method
        self.sampling_percent_uvp5 = sampling_percent_uvp5
        self.sampling_percent_uvp6 = sampling_percent_uvp6
        self.target_size = target_size
        self.test_dataset_sampling = test_dataset_sampling
        self.test_percent = test_percent


class TrainingConfig:
    def __init__(self, architecture_type, batch_size, gray, learning_rate, num_epoch):
        self.architecture_type = architecture_type
        self.batch_size = batch_size
        self.gray = gray
        self.learning_rate = learning_rate
        self.num_epoch = num_epoch


class PredictionConfig:
    def __init__(self, path_model, batch_size):
        self.path_model = path_model
        self.batch_size = batch_size


class Configuration:
    def __init__(self, config_file_path, input_path=None, output_path=None):
        with open(config_file_path, "r") as config_file:
            config_data = yaml.safe_load(config_file)

        self.input_path = input_path
        self.output_path = output_path
        self.base = BaseConfig(**config_data['base'])
        self.sampling = SamplingConfig(**config_data['sampling'])
        self.training = TrainingConfig(**config_data['training'])
        self.prediction = PredictionConfig(**config_data['prediction'])

    def write(self, filename):
        filename = Path(filename)
        if not filename.parent.exists():
            filename.parent.mkdir(parents=True)
        with filename.open("w") as file_handler:
            yaml.dump(
                self, file_handler, allow_unicode=True, default_flow_style=False
            )

