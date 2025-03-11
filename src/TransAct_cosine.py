import torch
from fuxictr.utils import not_in_whitelist
from torch import nn
import math
import random
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CrossNetV2
from fuxictr.pytorch.torch_utils import get_optimizer, get_loss
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineSchedule(_LRScheduler):
    """
    Learning rate scheduler combining linear warmup followed by cosine annealing.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for linear warmup.
        total_steps (int): Total number of training steps (warmup + annealing).
        base_lr (float): Maximum learning rate after warmup.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup: lr = base_lr * (step / warmup_steps)
            lr = self.base_lr * (self.last_epoch / max(1, self.warmup_steps))
        else:
            # Cosine annealing: lr = base_lr * 0.5*(1 + cos(pi * progress))
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp progress to [0, 1]
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = min(self.base_lr * cosine_decay, 1.e-6)

        return [lr for _ in self.base_lrs]


class TransAct_cosine(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="TransAct_cosine",
                 gpu=-1,
                 hidden_activations="ReLU",
                 dcn_cross_layers=3,
                 dcn_hidden_units=[256, 128, 64],
                 mlp_hidden_units=[],
                 num_heads=1,
                 transformer_layers=1,
                 transformer_dropout=0,
                 dim_feedforward=512,
                 learning_rate=1e-3,
                 embedding_dim=64,
                 net_dropout=0,
                 batch_norm=False,
                 first_k_cols=1,
                 use_time_window_mask=False,
                 time_window_ms=86400000,
                 concat_max_pool=True,
                 accumulation_steps=1,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 warmup_steps=500,
                 total_steps=5000,
                 base_lr=0.0048,
                 **kwargs):
        super().__init__(feature_map,
                         model_id=model_id,
                         gpu=gpu,
                         embedding_regularizer=embedding_regularizer,
                         net_regularizer=net_regularizer,
                         **kwargs)
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.item_info_dim = 0
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)

        transformer_in_dim = self.item_info_dim * 2

        self.accumulation_steps = accumulation_steps
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)

        self.transformer_encoder = TransActTransformer(
            transformer_in_dim,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            dropout=transformer_dropout,
            transformer_layers=transformer_layers,
            use_time_window_mask=use_time_window_mask,
            time_window_ms=time_window_ms,
            first_k_cols=first_k_cols,
            concat_max_pool=concat_max_pool
        )
        seq_out_dim = (first_k_cols + int(concat_max_pool)) * transformer_in_dim

        dcn_in_dim = feature_map.sum_emb_out_dim() + seq_out_dim
        self.crossnet = CrossNetV2(dcn_in_dim, dcn_cross_layers)
        self.parallel_dnn = MLP_Block(input_dim=dcn_in_dim,
                                      output_dim=None, # output hidden layer
                                      hidden_units=dcn_hidden_units,
                                      hidden_activations=hidden_activations,
                                      output_activation=None,
                                      dropout_rates=net_dropout,
                                      batch_norm=batch_norm)
        dcn_out_dim = dcn_in_dim + dcn_hidden_units[-1]
        self.mlp = MLP_Block(input_dim=dcn_out_dim,
                             output_dim=1,
                             hidden_units=mlp_hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.output_activation)

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def compile(self, optimizer, loss, lr):
        self.optimizer = get_optimizer(optimizer, self.parameters(), lr)
        self.loss_fn = get_loss(loss)
        self.scheduler = WarmupCosineSchedule(self.optimizer, self.warmup_steps, self.total_steps, self.base_lr)

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []
        if batch_dict: # not empty
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(feature_emb)
        feat_emb = torch.cat(emb_list, dim=-1)
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)

        target_emb = item_feat_emb[:, -1, :]
        sequence_emb = item_feat_emb[:, 0:-1, :]
        transact_emb = self.transformer_encoder(
                target_emb, sequence_emb, mask=mask
        )

        dcn_in_emb = torch.cat([feat_emb, target_emb, transact_emb], dim=-1)
        cross_out = self.crossnet(dcn_in_emb)
        dnn_out = self.parallel_dnn(dcn_in_emb)
        y_pred = self.mlp(torch.cat([cross_out, dnn_out], dim=-1))
        return_dict = {"y_pred": y_pred}
        return return_dict

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
        return X_dict, item_dict, mask.to(self.device)

    def concat_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

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
            self.scheduler.step()
        return loss


class TransActTransformer(nn.Module):
    def __init__(self,
                 transformer_in_dim,
                 dim_feedforward=64,
                 num_heads=1,
                 dropout=0,
                 transformer_layers=1,
                 use_time_window_mask=False,
                 time_window_ms=86400000, # recent 24h
                 first_k_cols=1,
                 concat_max_pool=True):
        super(TransActTransformer, self).__init__()
        self.use_time_window_mask = use_time_window_mask
        self.time_window_ms = time_window_ms
        self.concat_max_pool = concat_max_pool
        self.first_k_cols = first_k_cols
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_in_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        if self.concat_max_pool:
            self.out_linear = nn.Linear(transformer_in_dim, transformer_in_dim)

    def forward(self, target_emb, sequence_emb, time_interval_seq=None, mask=None):
        # concat action sequence emb with target emb
        seq_len = sequence_emb.size(1)
        concat_seq_emb = torch.cat([sequence_emb,
                                    target_emb.unsqueeze(1).expand(-1, seq_len, -1)], dim=-1)
        # get sequence mask (1's are masked)
        key_padding_mask = self.adjust_mask(mask).bool() # keep the last dim
        if self.use_time_window_mask and self.training:
            rand_time_window_ms = random.randint(0, self.time_window_ms)
            time_window_mask = (time_interval_seq < rand_time_window_ms)
            key_padding_mask = torch.bitwise_or(key_padding_mask, time_window_mask)
        tfmr_out = self.transformer_encoder(src=concat_seq_emb,
                                            src_key_padding_mask=key_padding_mask)
        tfmr_out = tfmr_out.masked_fill(
            key_padding_mask.unsqueeze(-1).repeat(1, 1, tfmr_out.shape[-1]), 0.
        )
        # process the transformer output
        output_concat = []
        output_concat.append(tfmr_out[:, -self.first_k_cols:].flatten(start_dim=1))
        if self.concat_max_pool:
            # Apply max pooling to the transformer output
            tfmr_out = tfmr_out.masked_fill(
                key_padding_mask.unsqueeze(-1).repeat(1, 1, tfmr_out.shape[-1]), -1e9
            )
            pooled_out = self.out_linear(tfmr_out.max(dim=1).values)
            output_concat.append(pooled_out)
        return torch.cat(output_concat, dim=-1)

    def adjust_mask(self, mask):
        # make sure not all actions in the sequence are masked
        fully_masked = mask.all(dim=-1)
        mask[fully_masked, -1] = 0
        return mask
