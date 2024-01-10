import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryAttentionModule(nn.Module):
    def __init__(self, query_size, memory_size, attention_units, num_dense_layers=1):
        super(MemoryAttentionModule, self).__init__()
        self.query_size = query_size
        self.memory_size = memory_size
        self.attention_units = attention_units
        self.num_dense_layers = num_dense_layers

        # Create a sequence of dense layers
        self.dense_layers = nn.ModuleList([
            nn.Linear(query_size if i == 0 else attention_units, attention_units)
            for i in range(num_dense_layers)
        ])

        self.softmax = nn.Softmax(dim=-1)
        self.normalize = nn.LayerNorm(memory_size)

    def forward(self, query, memory_keys):
        query_transformed = query

        # Apply multiple dense layers
        for layer in self.dense_layers:
            query_transformed = F.relu(layer(query_transformed))

        attention_scores = torch.matmul(query_transformed, memory_keys.transpose(0, 1))
        attention_weights = self.softmax(attention_scores)
        attended_memory = torch.matmul(attention_weights, memory_keys)
        output = self.normalize(attended_memory + query)
        return output


class MA(nn.Module):
    def __init__(self, model_name, weights_path, visual_encoder_size, query_size, memory_size, attention_units,
                 num_dense_layers, num_classes, k):
        super(MA, self).__init__()

        self.k = k

        # Visual encoder
        self.model_name = model_name
        self.weights_path = weights_path
        self.visual_encoder_size = visual_encoder_size
        self.visual_encoder = self.load_pretrained_visual_encoder()

        # Linear layer for classification
        self.classification_layer = nn.Linear(query_size, num_classes)

        # Memory Attention Module
        self.attention_units = attention_units
        self.num_dense_layers = num_dense_layers
        self.memory_attention = MemoryAttentionModule(query_size, memory_size, attention_units, num_dense_layers)

    def forward(self, visual_input, query_input, memory_keys):
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

    def load_pretrained_visual_encoder(self):
        if self.model_name == 'v_em1':
            model = nn.Linear(self.visual_encoder_size,
                              self.visual_encoder_size)
        elif self.model_name == 'v_em2':
            model = nn.Linear(self.visual_encoder_size,
                              self.visual_encoder_size)
        else:
            raise ValueError("Invalid model name")

        # Load pretrained weights
        if self.weights_path is not None:
            model.load_state_dict(torch.load(self.weights_path))

        # Freeze the parameters
        for param in model.parameters():
            param.requires_grad = False

        return model

    @staticmethod
    def knn_search(query, memory_keys, k):
        # Calculate cosine similarity between query and memory keys
        similarity_scores = F.cosine_similarity(query.unsqueeze(0), memory_keys, dim=1)

        # Get indices of top-k similar memory keys
        _, indices = torch.topk(similarity_scores, k, dim=0)

        # Gather the top-k memory keys
        memory_keys_knn = memory_keys[indices]

        return memory_keys_knn


