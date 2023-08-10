import yaml


class SamplingConfig:
    def __init__(self, path_uvp5, path_uvp6, path_output, uvp_type, num_class,
                 sampling_method, sampling_percent_uvp5, sampling_percent_uvp6, target_size):
        self.path_uvp5 = path_uvp5
        self.path_uvp6 = path_uvp6
        self.path_output = path_output
        self.uvp_type = uvp_type
        self.num_class = num_class
        self.sampling_method = sampling_method
        self.sampling_percent_uvp5 = sampling_percent_uvp5
        self.sampling_percent_uvp6 = sampling_percent_uvp6
        self.target_size = target_size


class TrainingConfig:
    def __init__(self, architecture_type, batch_size):
        self.architecture_type = architecture_type
        self.batch_size = batch_size


class Configuration:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as config_file:
            config_data = yaml.safe_load(config_file)

        self.sampling = SamplingConfig(**config_data['sampling'])
        self.training = TrainingConfig(**config_data['training'])