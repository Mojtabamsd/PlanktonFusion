import torch
import torch.nn as nn
import torch.nn.functional as F
from models.autoencoder import ConvAutoencoder, ResNetCustom
from pathlib import Path


class MemoryAttentionModule(nn.Module):
    def __init__(self, query_size, memory_size, attention_units, num_dense_layers=1):
        super(MemoryAttentionModule, self).__init__()
        self.query_size = query_size
        self.memory_size = memory_size
        self.attention_units = attention_units
        self.num_dense_layers = num_dense_layers

        # Dense layers for both queries and memory keys
        self.dense_layers_query = nn.ModuleList([
            nn.Linear(query_size if i == 0 else attention_units, attention_units)
            for i in range(num_dense_layers)
        ])

        self.dense_layers_memory = nn.ModuleList([
            nn.Linear(memory_size if i == 0 else attention_units, attention_units)
            for i in range(num_dense_layers)
        ])

        self.softmax = nn.Softmax(dim=-1)
        self.normalize = nn.LayerNorm(attention_units)

    def forward(self, query, memory_keys):

        batch_size, k, _ = memory_keys.size()

        query_transformed = query
        for layer in self.dense_layers_query:
            query_transformed = F.relu(layer(query_transformed))

        memory_keys_transformed = memory_keys
        for layer in self.dense_layers_memory:
            memory_keys_transformed = F.relu(layer(memory_keys_transformed))

        query_transformed_expand = query_transformed.unsqueeze(1).expand(-1, k, -1)
        attention_scores = torch.sum(query_transformed_expand * memory_keys_transformed, dim=-1, keepdim=True)

        attention_weights = self.softmax(attention_scores)
        attended_memory = torch.sum(attention_weights * memory_keys_transformed, dim=1)
        output = self.normalize(attended_memory + query_transformed)
        return output


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

        # Linear layer for classification
        self.classification_layer = nn.Linear(self.query_size + self.attention_units, self.num_classes)

        # Memory Attention Module
        self.memory_attention = MemoryAttentionModule(self.query_size, self.memory_size, self.attention_units,
                                                      self.num_dense_layers)

    def forward(self, query_input, memory_keys):
        # Visual encoding for query
        query_embedding = F.relu(self.visual_encoder(query_input))

        # k-NN search
        memory_keys_knn = self.knn_search(query_embedding, memory_keys, self.k)

        # Apply Memory Attention Module
        memory_attention_output = self.memory_attention(query_embedding, memory_keys_knn)

        # Concatenate query_embedding and memory_attention_output
        merged_output = torch.cat([query_embedding, memory_attention_output], dim=1)

        # Classification layer
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


