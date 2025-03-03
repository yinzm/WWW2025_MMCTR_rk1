import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, DIN_Attention, Dice
from fuxictr.pytorch.torch_utils import get_activation
from fuxictr.utils import not_in_whitelist


class PPNet(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="PPNet", 
                 gpu=-1,
                 attention_hidden_units=[64],
                 attention_hidden_activations="Dice",
                 attention_output_activation=None,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10,
                 gate_emb_dim=10,
                 gate_priors=[],
                 gate_hidden_dim=64,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 attention_use_softmax=False,
                 accumulation_steps=1,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        super(PPNet, self).__init__(feature_map, 
                                    model_id=model_id, 
                                    gpu=gpu, 
                                    embedding_regularizer=embedding_regularizer, 
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        self.feature_map = feature_map
        self.gate_priors = gate_priors
        self.embedding_dim = embedding_dim
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)
        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.gate_embed_layer = FeatureEmbedding(feature_map, gate_emb_dim, 
                                                 required_feature_columns=gate_priors)
        input_dim = feature_map.sum_emb_out_dim() + self.item_info_dim
        gate_input_dim = input_dim + len(gate_priors) * gate_emb_dim

        self.attention_layers = DIN_Attention(
            self.item_info_dim,
            attention_units=attention_hidden_units,
            hidden_activations=attention_hidden_activations,
            output_activation=attention_output_activation,
            dropout_rate=attention_dropout,
            use_softmax=attention_use_softmax
        )

        self.ppn = PPNet_MLP(input_dim=input_dim,
                             output_dim=1,
                             gate_input_dim=gate_input_dim,
                             gate_hidden_dim=gate_hidden_dim,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()
            
    def forward(self, inputs):
        batch_dict, item_dict, gate_dict, mask = self.get_inputs(inputs, self.gate_priors)

        emb_list = []
        if batch_dict: # not empty
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(feature_emb)

        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)
        target_emb = item_feat_emb[:, -1, :]
        sequence_emb = item_feat_emb[:, 0:-1, :]

        pooling_emb = self.attention_layers(target_emb, sequence_emb, mask)
        emb_list += [target_emb, pooling_emb]
        feature_emb = torch.cat(emb_list, dim=-1)

        gate_emb = self.gate_embed_layer(gate_dict, flatten_emb=True)
        y_pred = self.ppn(feature_emb, gate_emb)
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_inputs(self, inputs, gate_priors, feature_source=None):
        batch_dict, item_dict, mask = inputs
        batch_size = mask.shape[0]
        X_dict = dict()
        gate_dict = dict()
        for feature, value in batch_dict.items():
            if feature in gate_priors:
                gate_dict[feature] = value.to(self.device)
            if feature in self.feature_map.labels:
                continue
            feature_spec = self.feature_map.features[feature]
            if feature_spec["type"] == "meta":
                continue
            if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
                continue
            X_dict[feature] = value.to(self.device)
        for item, value in item_dict.items():
            item_dict[item] = value.to(self.device)
            if item in gate_priors:
                gate_dict[item] = value.view(batch_size, -1)[:, -1].to(self.device)
        return X_dict, item_dict, gate_dict, mask.to(self.device)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)

    def train_step(self, batch_data):
        return_dict = self.forward(batch_data)
        y_true = self.get_labels(batch_data)
        loss = self.compute_loss(return_dict, y_true)
        loss = loss / self.accumulation_steps
        loss.backward()
        if (self._batch_index + 1) % self.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss


class PPNet_MLP(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim=1,
                 gate_input_dim=64,
                 gate_hidden_dim=None,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 dropout_rates=0.0,
                 batch_norm=False,
                 use_bias=True):
        super(PPNet_MLP, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        self.gate_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            layers = [nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx] is not None:
                layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                layers.append(nn.Dropout(p=dropout_rates[idx]))
            self.mlp_layers.append(nn.Sequential(*layers))
            self.gate_layers.append(GateNU(gate_input_dim, gate_hidden_dim, 
                                           output_dim=hidden_units[idx + 1]))
        self.mlp_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
    
    def forward(self, feature_emb, gate_emb):
        gate_input = torch.cat([feature_emb.detach(), gate_emb], dim=-1)
        h = feature_emb
        for i in range(len(self.gate_layers)):
            h = self.mlp_layers[i](h)
            g = self.gate_layers[i](gate_input)
            h = h * g
        out = self.mlp_layers[-1](h)
        return out


class GateNU(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim=None,
                 output_dim=None,
                 hidden_activation="ReLU",
                 dropout_rate=0.0):
        super(GateNU, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        layers = [nn.Linear(input_dim, hidden_dim)]
        layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.gate = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.gate(inputs) * 2
