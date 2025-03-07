import torch
from fuxictr.utils import not_in_whitelist
from torch import nn
import numpy as np
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, FeatureEmbedding
from torch.nn import MultiheadAttention


class BST(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="BST", 
                 gpu=-1, 
                 dnn_hidden_units=[256, 128, 64],
                 dnn_activations="ReLU",
                 num_heads=2,
                 ori_sql_len=64,
                 stacked_transformer_layers=1,
                 attention_dropout=0,
                 learning_rate=1e-3, 
                 embedding_dim=10,
                 net_dropout=0,
                 batch_norm=False,
                 layer_norm=True,
                 use_residual=True,
                 seq_pooling_type="mean", # ["mean", "sum", "target", "concat"]
                 use_position_emb=True,
                 use_causal_mask=False,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(BST, self).__init__(feature_map, 
                                  model_id=model_id, 
                                  gpu=gpu, 
                                  embedding_regularizer=embedding_regularizer, 
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.use_causal_mask = use_causal_mask
        self.seq_pooling_type = seq_pooling_type
        self.feature_map = feature_map
        self.accumulation_steps = accumulation_steps
        self.embedding_dim = embedding_dim
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)

        self.num_heads = num_heads
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        model_dim = embedding_dim * (4 + int(use_position_emb))
        seq_len = ori_sql_len + 1
        seq_out_dim = self.get_seq_out_dim(model_dim, seq_len)

        self.transformer_encoder = BehaviorTransformer(seq_len=seq_len,
                            model_dim=model_dim,
                            num_heads=num_heads,
                            stacked_transformer_layers=stacked_transformer_layers,
                            attn_dropout=attention_dropout,
                            net_dropout=net_dropout,
                            position_dim=embedding_dim,
                            use_position_emb=use_position_emb,
                            layer_norm=layer_norm,
                            use_residual=use_residual)
        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim() + seq_out_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def get_seq_out_dim(self, model_dim, seq_len):
        if self.seq_pooling_type == "concat":
            seq_out_dim = seq_len * model_dim
        else:
            seq_out_dim = model_dim
        return seq_out_dim
        
    def forward(self, inputs):
        batch_dict, item_dict, padding_mask, attn_mask = self.get_inputs(inputs)
        emb_list = []
        if batch_dict: # not empty
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(feature_emb)
        feat_emb = torch.cat(emb_list, dim=-1)
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = padding_mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_feat_emb[:, -1, :]
        sequence_emb = item_feat_emb[:, 0:-1, :]
        concat_seq_emb = torch.cat([sequence_emb, target_emb.unsqueeze(1)], dim=1)
        transformer_out = self.transformer_encoder(concat_seq_emb, attn_mask) # b x len x emb
        pooling_emb = self.sequence_pooling(transformer_out, padding_mask)

        y_pred = self.dnn(torch.cat([feat_emb, target_emb, pooling_emb], dim=-1))
        return_dict = {"y_pred": y_pred}
        return return_dict

    def get_mask(self, x):
        """ padding_mask: B x L, 1 for masked positions
            attn_mask: (B*H) x L x L, 1 for masked positions in nn.MultiheadAttention
        """
        padding_mask = (x == 0)
        padding_mask = torch.cat([padding_mask, torch.zeros(x.size(0), 1).bool().to(x.device)],
                                 dim=-1)
        seq_len = padding_mask.size(1)
        attn_mask = padding_mask.unsqueeze(1).repeat(1, seq_len, 1)
        diag_zeros = ~torch.eye(seq_len, device=x.device).bool().unsqueeze(0).expand_as(attn_mask)
        attn_mask = attn_mask & diag_zeros
        if self.use_causal_mask:
            causal_mask = (
                torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1)
                .bool().unsqueeze(0).expand_as(attn_mask)
            )
            attn_mask = attn_mask | causal_mask
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(end_dim=1)
        return padding_mask, attn_mask

    def sequence_pooling(self, transformer_out, mask):
        mask = (1 - mask.float()).unsqueeze(-1) # 0 for masked positions
        if self.seq_pooling_type == "mean":
            return (transformer_out * mask).sum(dim=1) / (mask.sum(dim=1) + 1.e-12)
        elif self.seq_pooling_type == "sum":
            return (transformer_out * mask).sum(dim=1)
        elif self.seq_pooling_type == "target":
            return transformer_out[:, -1, :]
        elif self.seq_pooling_type == "concat":
            return transformer_out.flatten(start_dim=1)
        else:
            raise ValueError("seq_pooling_type={} not supported.".format(self.seq_pooling_type))

    def concat_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

    def get_inputs(self, inputs, feature_source=None):
        batch_dict, item_dict, mask = inputs
        X_dict = dict()
        for feature, value in batch_dict.items():
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
        batch_size = mask.shape[0]
        item_seq = item_dict['item_id'].view(batch_size, -1)[:, 0:-1]
        padding_mask, attn_mask = self.get_mask(item_seq)

        return X_dict, item_dict, padding_mask, attn_mask

    def get_labels(self, inputs):
        """ Please override get_labels() when using multiple labels!
        """
        labels = self.feature_map.labels
        batch_dict = inputs[0]
        y = batch_dict[labels[0]].to(self.device)
        return y.float().view(-1, 1)

    def get_group_id(self, inputs):
        return inputs[0][self.feature_map.group_id]

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


class BehaviorTransformer(nn.Module):
    def __init__(self,
                 seq_len=1,
                 model_dim=64,
                 num_heads=8,
                 stacked_transformer_layers=1,
                 attn_dropout=0.0,
                 net_dropout=0.0,
                 use_position_emb=True,
                 position_dim=4,
                 layer_norm=True,
                 use_residual=True):
        super(BehaviorTransformer, self).__init__()
        self.position_dim = position_dim
        self.use_position_emb = use_position_emb
        self.transformer_blocks = nn.ModuleList(TransformerBlock(model_dim=model_dim,
                                                                 ffn_dim=model_dim,
                                                                 num_heads=num_heads, 
                                                                 attn_dropout=attn_dropout, 
                                                                 net_dropout=net_dropout,
                                                                 layer_norm=layer_norm,
                                                                 use_residual=use_residual)
                                                for _ in range(stacked_transformer_layers))
        if self.use_position_emb:
            self.position_emb = nn.Parameter(torch.Tensor(seq_len, position_dim))
            self.reset_parameters()

    def reset_parameters(self):
        seq_len = self.position_emb.size(0)
        pe = torch.zeros(seq_len, self.position_dim)
        position = torch.arange(0, seq_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.position_dim, 2).float() * (-np.log(10000.0) / self.position_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.position_emb.data = pe

    def forward(self, x, attn_mask=None):
        # input b x len x dim
        if self.use_position_emb:
            x = torch.cat([x, self.position_emb.unsqueeze(0).repeat(x.size(0), 1, 1)], dim=-1)
        for i in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[i](x, attn_mask=attn_mask)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, model_dim=64, ffn_dim=64, num_heads=8, attn_dropout=0.0, net_dropout=0.0,
                 layer_norm=True, use_residual=True):
        super(TransformerBlock, self).__init__()
        self.attention = MultiheadAttention(model_dim,
                                            num_heads=num_heads, 
                                            dropout=attn_dropout,
                                            batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(model_dim, ffn_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(ffn_dim, model_dim))
        self.use_residual = use_residual
        self.dropout1 = nn.Dropout(net_dropout)
        self.dropout2 = nn.Dropout(net_dropout)
        self.layer_norm1 = nn.LayerNorm(model_dim) if layer_norm else None
        self.layer_norm2 = nn.LayerNorm(model_dim) if layer_norm else None

    def forward(self, x, attn_mask=None):
        attn, _ = self.attention(x, x, x, attn_mask=attn_mask)
        s = self.dropout1(attn)
        if self.use_residual:
            s += x
        if self.layer_norm1 is not None:
            s = self.layer_norm1(s)
        out = self.dropout2(self.ffn(s))
        if self.use_residual:
            out += s
        if self.layer_norm2 is not None:
            out = self.layer_norm2(out)
        return out
