import torch
import torch.nn as nn
import torch.nn.functional as F
from models.autoencoder import ConvAutoencoder, ResNetCustom
from pathlib import Path


class MemoryAttentionModule(nn.Module):
    def __init__(self, query_size, memory_size, attention_units):
        super(MemoryAttentionModule, self).__init__()
        self.query_size = query_size
        self.memory_size = memory_size
        self.attention_units = attention_units

        self.query_dense = nn.Linear(query_size, attention_units)
        self.memory_dense = nn.Linear(memory_size, attention_units)
        self.attention_score_dense = nn.Linear(attention_units, 1)

    def forward(self, query, memory_keys):
        expanded_query = query.unsqueeze(1)
        attention_scores = self.attention_score_dense(torch.tanh(self.query_dense(expanded_query)
                                                                 + self.memory_dense(memory_keys)))

        attention_weights = F.softmax(attention_scores, dim=1)

        attended_memory = torch.sum(attention_weights * memory_keys, dim=1)

        return attended_memory


class MA(nn.Module):
    def __init__(self, config, console):
        super(MA, self).__init__()

        self.model_name = config.autoencoder.architecture_type
        self.weights_path = config.memory.visual_embedded_model
        self.visual_encoder_size = config.autoencoder.latent_dim
        self.query_size = config.autoencoder.latent_dim
        self.memory_size = config.autoencoder.latent_dim
        self.attention_units = 256
        self.num_dense_layers = 1
        self.num_classes = config.sampling.num_class
        self.k = config.memory.k
        self.input_size = config.sampling.target_size
        self.gray = config.autoencoder.gray
        self.device = config.device

        # Visual encoder
        self.visual_encoder = self.load_pretrained_visual_encoder(console)

        # Memory Attention Module
        self.memory_attention = MemoryAttentionModule(self.query_size, self.memory_size, self.attention_units)

        self.classification_layer = nn.Linear(2 * self.query_size, self.num_classes)

    def forward(self, query_input, memory_keys):
        query_embedding = F.relu(self.visual_encoder(query_input))

        # k-NN search
        memory_keys_knn = self.knn_search(query_embedding, memory_keys, self.k)

        memory_attention_output = self.memory_attention(query_embedding, memory_keys_knn)
        merged_output = torch.cat([query_embedding, memory_attention_output], dim=1)

        output = self.classification_layer(merged_output)
        return output

    def load_pretrained_visual_encoder(self, console):
        if self.model_name == 'conv_autoencoder':
            model = ConvAutoencoder(latent_dim=self.visual_encoder_size,
                                    input_size=self.input_size,
                                    gray=self.gray,
                                    encoder_mode=True)
        elif self.model_name == 'resnet18':
            model = ResNetCustom(num_classes=self.num_classes,
                                 latent_dim=self.visual_encoder_size,
                                 gray=self.gray,
                                 encoder_mode=True)
        else:
            raise ValueError("Invalid visual model name")

        # Load pretrained weights
        if self.weights_path is not None:
            saved_weights = "model_weights_final.pth"
            training_path = Path(self.weights_path)
            saved_weights_file = training_path / saved_weights

            console.info("Model loaded from ", saved_weights_file)
            model.load_state_dict(torch.load(saved_weights_file, map_location=self.device))
            model.to(self.device)

        # Freeze the parameters
        for param in model.parameters():
            param.requires_grad = False

        return model

    @staticmethod
    def knn_search(queries, memory_keys, k):
        # Calculate cosine similarity between queries and memory keys
        similarity_scores = F.cosine_similarity(queries.unsqueeze(1), memory_keys, dim=2)

        # Get indices of top-k similar memory keys for each query
        _, indices = torch.topk(similarity_scores, k, dim=1)

        # Gather the top-k memory keys for each query
        memory_keys_knn = torch.gather(memory_keys.unsqueeze(0).expand(queries.size(0), -1, -1), 1,
                                       indices.unsqueeze(2).expand(-1, -1, memory_keys.size(1)))

        return memory_keys_knn


