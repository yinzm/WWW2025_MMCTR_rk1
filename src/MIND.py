# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from fuxictr.pytorch.models import BaseModel
# from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
# from fuxictr.utils import not_in_whitelist

# # 定义 sequence_mask 函数，根据有效长度生成 mask
# def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
#     if maxlen is None:
#         maxlen = lengths.max()
#     # 在相同设备上生成 row_vector
#     row_vector = torch.arange(0, maxlen, device=lengths.device)
#     matrix = lengths.unsqueeze(-1)
#     mask = row_vector < matrix
#     return mask.to(dtype)

# # 定义 squash 激活函数，将向量长度压缩到 (0, 1) 之间
# def squash(inputs, epsilon=1e-9):
#     vec_squared_norm = torch.sum(torch.square(inputs), dim=-1, keepdim=True)
#     scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + epsilon)
#     return scalar_factor * inputs

# # 封装动态路由模块
# class Routing(nn.Module):
#     def __init__(self, seq_len, input_dim, num_capsules, routing_iters=3):
#         """
#         Args:
#             seq_len: 历史行为序列的最大长度
#             input_dim: 每个行为嵌入的维度
#             num_capsules: capsule 数量（即提取的兴趣数目）
#             routing_iters: 动态路由迭代次数
#         """
#         super(Routing, self).__init__()
#         self.routing_iters = routing_iters
#         self.num_capsules = num_capsules
#         self.seq_len = seq_len
#         self.input_dim = input_dim
#         # 初始化路由 logits B，形状为 (1, num_capsules, seq_len)，设置为不可训练
#         self.register_buffer("B_matrix", torch.randn(1, num_capsules, seq_len))
#         # 初始化线性变换矩阵 S，形状为 (input_dim, input_dim)
#         self.S_matrix = nn.Parameter(torch.randn(input_dim, input_dim))

#     def forward(self, low_capsule, seq_len_tensor):
#         """
#         Args:
#             low_capsule: [batch_size, seq_len, input_dim] 低层 capsule 表示
#             seq_len_tensor: [batch_size, 1] 每个样本有效序列的长度
#         Returns:
#             capsule: [batch_size, num_capsules, input_dim] 高层 capsule 表示
#         """
#         batch_size = low_capsule.size(0)
#         # 扩展 B_matrix 为 [batch_size, num_capsules, seq_len]
#         B = self.B_matrix.expand(batch_size, -1, -1)
#         # 将 seq_len_tensor 扩展为 [batch_size, num_capsules]
#         seq_len_tile = seq_len_tensor.repeat(1, self.num_capsules)
#         for i in range(self.routing_iters):
#             # 根据有效长度生成 mask，形状为 [batch_size, num_capsules, seq_len]
#             mask = sequence_mask(seq_len_tile, self.seq_len, dtype=torch.bool)
#             # 将无效位置替换为一个很小的值，保证 softmax 时该位置权重趋近于0
#             pad = torch.ones_like(mask, dtype=torch.float32) * (-2 ** 16 + 1)
#             B_masked = torch.where(mask, B, pad)
#             # 对 B_masked 在 seq_len 维度上做 softmax 得到路由权重 W
#             W = F.softmax(B_masked, dim=-1)
#             # 对低层 capsule 进行线性变换，使用 einsum 高效计算： [B, seq_len, input_dim] -> [B, seq_len, input_dim]
#             low_capsule_trans = torch.einsum('bij,jk->bik', low_capsule, self.S_matrix)
#             # 根据路由权重 W 对低层 capsule 进行加权求和，得到 capsule 表示
#             capsule_temp = torch.matmul(W, low_capsule_trans)
#             capsule = squash(capsule_temp)
#             if i < self.routing_iters - 1:
#                 # 计算 capsule 与低层 capsule 转换后表示的相似度（agreement）
#                 agreement = torch.matmul(capsule, low_capsule_trans.transpose(1, 2))
#                 # 更新路由 logits B（累加 agreement）
#                 B = B + agreement
#         return capsule

# # 修改后的 MultiInterestExtractor，使用独立的 Routing 模块实现动态路由
# class MultiInterestExtractor(nn.Module):
#     def __init__(self, input_dim, num_capsules=4, routing_iters=3, seq_len=None):
#         """
#         Args:
#             input_dim: 每个历史行为的嵌入维度。
#             num_capsules: 提取的兴趣数目。
#             routing_iters: 动态路由迭代次数。
#             seq_len: 历史行为序列的最大长度（用于初始化路由系数）。
#         """
#         super(MultiInterestExtractor, self).__init__()
#         if seq_len is None:
#             raise ValueError("seq_len must be provided")
#         self.num_capsules = num_capsules
#         self.routing_iters = routing_iters
#         self.seq_len = seq_len
#         # 实例化动态路由模块
#         self.routing = Routing(seq_len, input_dim, num_capsules, routing_iters)
#         # 两层全连接（MLP）对 capsule 表示进行进一步变换
#         self.dense1 = nn.Linear(input_dim, 4 * input_dim)
#         self.dense2 = nn.Linear(4 * input_dim, input_dim)

#     def forward(self, sequence_emb, mask):
#         """
#         Args:
#             sequence_emb: [batch_size, L, input_dim] 用户历史行为嵌入。
#             mask: [batch_size, L] 布尔型张量，指示有效行为位置。
#         Returns:
#             capsules: [batch_size, num_capsules, input_dim] 多兴趣表示。
#         """
#         bs, L, D = sequence_emb.shape
#         # 若 mask 长度与实际序列长度不一致，则做相应调整
#         if mask.size(1) != L:
#             if mask.size(1) < L:
#                 pad_size = L - mask.size(1)
#                 pad_mask = torch.zeros(bs, pad_size, dtype=mask.dtype, device=mask.device)
#                 mask = torch.cat([mask, pad_mask], dim=1)
#             else:
#                 mask = mask[:, :L]
#         # 根据 mask 计算每个样本的有效行为数，形状为 [batch_size, 1]
#         seq_lens = mask.sum(dim=1, keepdim=True)
#         # 调用动态路由模块得到 capsule 表示
#         capsules = self.routing(sequence_emb, seq_lens)
#         # 经过两层 MLP 进一步变换 capsule 表示
#         capsules = self.dense2(F.relu(self.dense1(capsules)))
#         return capsules

# # 修改后的 MIND 模型，保持其他模块结构不变，仅更新多兴趣提取部分
# class MIND(BaseModel):
#     def __init__(self, feature_map, 
#                  dnn_hidden_units=[512, 128, 64],
#                  dnn_activations="ReLU",
#                  num_capsules=4,
#                  routing_iters=3,
#                  seq_len=65,  # 默认历史行为序列长度，根据实际情况调整
#                  learning_rate=1e-3, 
#                  embedding_dim=10, 
#                  net_dropout=0, 
#                  batch_norm=False, 
#                  accumulation_steps=1,
#                  embedding_regularizer=None, 
#                  net_regularizer=None,
#                  **kwargs):
#         kwargs.pop("model_id", None)
#         kwargs.pop("gpu", None)
#         super(MIND, self).__init__(feature_map,
#                                    model_id="MIND", 
#                                    gpu=kwargs.get("gpu", -1), 
#                                    embedding_regularizer=embedding_regularizer, 
#                                    net_regularizer=net_regularizer,
#                                    **kwargs)
#         self.accumulation_steps = accumulation_steps
#         self.feature_map = feature_map
#         self.embedding_dim = embedding_dim
#         self.item_info_dim = 0
#         # 统计所有物品相关特征的嵌入维度（只统计 item 特征）
#         for feat, spec in self.feature_map.features.items():
#             if spec.get("source") == "item":
#                 self.item_info_dim += spec.get("embedding_dim", embedding_dim)
        
#         self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
#         self.multi_interest_extractor = MultiInterestExtractor(
#             input_dim=self.item_info_dim,
#             num_capsules=num_capsules,
#             routing_iters=routing_iters,
#             seq_len=seq_len
#         )
        
#         # 修改 _calculate_input_dim：不将 item 特征重复计入 non_meta_emb_dim
#         input_dim = self._calculate_input_dim(feature_map, embedding_dim, num_capsules, self.item_info_dim)
#         self.dnn = MLP_Block(input_dim=input_dim,
#                              output_dim=1,
#                              hidden_units=dnn_hidden_units,
#                              hidden_activations=dnn_activations,
#                              output_activation=self.output_activation, 
#                              dropout_rates=net_dropout,
#                              batch_norm=batch_norm)
#         self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
#         self.reset_parameters()
#         self.model_to_device()

#     def _calculate_input_dim(self, feature_map, embedding_dim, num_capsules, item_info_dim):
#         """
#         计算 DNN 的输入维度，只包含非物品特征和经过多兴趣提取后的 capsule 表示（扁平化后）。
#         """
#         non_meta_emb_dim = 0
#         for feat, spec in feature_map.features.items():
#             if feat in feature_map.labels:
#                 continue
#             if spec.get("type") == "meta":
#                 continue
#             # 排除物品特征，避免重复计入
#             if spec.get("source") == "item":
#                 continue
#             non_meta_emb_dim += spec.get("embedding_dim", embedding_dim)
#         return non_meta_emb_dim + num_capsules * item_info_dim

#     def get_inputs(self, inputs, feature_source=None):
#         """
#         根据 inputs 类型提取 batch_dict、item_dict 和 mask。
#         """
#         if isinstance(inputs, tuple):
#             batch_dict, item_dict, mask = inputs
#         elif isinstance(inputs, dict):
#             batch_dict = inputs.get("batch_dict", {})
#             item_dict = inputs.get("item_dict", {})
#             mask = inputs.get("mask", None)
#         else:
#             raise ValueError("Inputs must be either a tuple or a dict.")
        
#         X_dict = {}
#         for feature, value in batch_dict.items():
#             if feature in self.feature_map.labels:
#                 continue
#             feature_spec = self.feature_map.features[feature]
#             if feature_spec["type"] == "meta":
#                 continue
#             if feature_source and not_in_whitelist(feature_spec["source"], feature_source):
#                 continue
#             X_dict[feature] = value.to(self.device)
        
#         for item, value in item_dict.items():
#             item_dict[item] = value.to(self.device)
        
#         if mask is not None:
#             mask = mask.to(self.device)
        
#         return X_dict, item_dict, mask

#     def get_group_id(self, inputs):
#         """
#         重写 get_group_id 方法，处理 tuple 格式的输入，
#         假设 group_id 存储在 batch_dict 中，且对应键为 self.feature_map.group_id
#         """
#         if isinstance(inputs, tuple):
#             batch_dict = inputs[0]
#         elif isinstance(inputs, dict):
#             batch_dict = inputs
#         else:
#             raise ValueError("Unsupported input type for get_group_id")
#         return batch_dict[self.feature_map.group_id]

#     def forward(self, inputs):
#         batch_dict, item_dict, mask = self.get_inputs(inputs)
#         emb_list = []
        
#         # 用户侧普通特征嵌入
#         if batch_dict:
#             feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
#             emb_list.append(feature_emb)
        
#         # 物品或历史行为嵌入
#         item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
#         batch_size = mask.shape[0]
#         # 假设 item_feat_emb 原始形状为 (batch_size, seq_len * item_info_dim)，重塑为 (batch_size, seq_len, item_info_dim)
#         item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)
        
#         # 利用动态路由提取多兴趣表示
#         multi_interest = self.multi_interest_extractor(item_feat_emb, mask)
#         multi_interest_flat = multi_interest.view(batch_size, -1)
#         emb_list.append(multi_interest_flat)
        
#         # 拼接所有嵌入作为 DNN 的输入，并返回 {"y_pred": y_pred}
#         feature_emb = torch.cat(emb_list, dim=-1)
#         y_pred = self.dnn(feature_emb)
#         return {"y_pred": y_pred}

#     def get_labels(self, inputs):
#         """
#         重写 get_labels 方法，处理 tuple 格式的输入，并确保返回的标签形状与预测一致，
#         且 dtype 为 Float。
#         假设 inputs 为 (batch_dict, item_dict, mask)，标签存储在 batch_dict 中。
#         """
#         if isinstance(inputs, tuple):
#             batch_dict = inputs[0]
#             labels = self.feature_map.labels
#             y = batch_dict[labels[0]].to(self.device)
#         elif isinstance(inputs, dict):
#             labels = self.feature_map.labels
#             y = inputs[labels[0]].to(self.device)
#         else:
#             raise ValueError("Unsupported input type for get_labels")
        
#         # 如果标签是一维的，则扩展最后一个维度
#         if y.dim() == 1:
#             y = y.unsqueeze(-1)
#         return y.float()

#     def train_step(self, batch_data):
#         return_dict = self.forward(batch_data)
#         y_true = self.get_labels(batch_data)
#         loss = self.compute_loss(return_dict, y_true)
#         loss = loss / self.accumulation_steps
#         loss.backward()
#         if (self._batch_index + 1) % self.accumulation_steps == 0:
#             nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
#             self.optimizer.step()
#             self.optimizer.zero_grad()
#         self._batch_index += 1
#         return loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.utils import not_in_whitelist

# 定义 sequence_mask 函数，根据有效长度生成 mask
def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    # 在相同设备上生成 row_vector
    row_vector = torch.arange(0, maxlen, device=lengths.device)
    matrix = lengths.unsqueeze(-1)
    mask = row_vector < matrix
    return mask.to(dtype)

# 定义 squash 激活函数，将向量长度压缩到 (0, 1) 之间
def squash(inputs, epsilon=1e-9):
    vec_squared_norm = torch.sum(torch.square(inputs), dim=-1, keepdim=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / torch.sqrt(vec_squared_norm + epsilon)
    return scalar_factor * inputs

# 封装动态路由模块
class Routing(nn.Module):
    def __init__(self, seq_len, input_dim, num_capsules, routing_iters=3):
        """
        Args:
            seq_len: 历史行为序列的最大长度
            input_dim: 每个行为嵌入的维度
            num_capsules: capsule 数量（即提取的兴趣数目）
            routing_iters: 动态路由迭代次数
        """
        super(Routing, self).__init__()
        self.routing_iters = routing_iters
        self.num_capsules = num_capsules
        self.seq_len = seq_len
        self.input_dim = input_dim
        # 初始化路由 logits B，形状为 (1, num_capsules, seq_len)，设置为不可训练
        self.register_buffer("B_matrix", torch.randn(1, num_capsules, seq_len))
        # 初始化线性变换矩阵 S，形状为 (input_dim, input_dim)
        self.S_matrix = nn.Parameter(torch.randn(input_dim, input_dim))

    def forward(self, low_capsule, seq_len_tensor):
        """
        Args:
            low_capsule: [batch_size, seq_len, input_dim] 低层 capsule 表示
            seq_len_tensor: [batch_size, 1] 每个样本有效序列的长度
        Returns:
            capsule: [batch_size, num_capsules, input_dim] 高层 capsule 表示
        """
        batch_size = low_capsule.size(0)
        # 扩展 B_matrix 为 [batch_size, num_capsules, seq_len]
        B = self.B_matrix.expand(batch_size, -1, -1)
        # 将 seq_len_tensor 扩展为 [batch_size, num_capsules]
        seq_len_tile = seq_len_tensor.repeat(1, self.num_capsules)
        for i in range(self.routing_iters):
            # 根据有效长度生成 mask，形状为 [batch_size, num_capsules, seq_len]
            mask = sequence_mask(seq_len_tile, self.seq_len, dtype=torch.bool)
            # 将无效位置替换为一个很小的值，保证 softmax 时该位置权重趋近于0
            pad = torch.ones_like(mask, dtype=torch.float32) * (-2 ** 16 + 1)
            B_masked = torch.where(mask, B, pad)
            # 对 B_masked 在 seq_len 维度上做 softmax 得到路由权重 W
            W = F.softmax(B_masked, dim=-1)
            # 对低层 capsule 进行线性变换，使用 einsum 高效计算： [B, seq_len, input_dim] -> [B, seq_len, input_dim]
            low_capsule_trans = torch.einsum('bij,jk->bik', low_capsule, self.S_matrix)
            # 根据路由权重 W 对低层 capsule 进行加权求和，得到 capsule 表示
            capsule_temp = torch.matmul(W, low_capsule_trans)
            capsule = squash(capsule_temp)
            if i < self.routing_iters - 1:
                # 计算 capsule 与低层 capsule 转换后表示的相似度（agreement）
                agreement = torch.matmul(capsule, low_capsule_trans.transpose(1, 2))
                # 更新路由 logits B（累加 agreement）
                B = B + agreement
        return capsule

# 修改后的 MultiInterestExtractor，使用独立的 Routing 模块实现动态路由
class MultiInterestExtractor(nn.Module):
    def __init__(self, input_dim, num_capsules=4, routing_iters=3, seq_len=None):
        """
        Args:
            input_dim: 每个历史行为的嵌入维度。
            num_capsules: 提取的兴趣数目。
            routing_iters: 动态路由迭代次数。
            seq_len: 历史行为序列的最大长度（用于初始化路由系数）。
        """
        super(MultiInterestExtractor, self).__init__()
        if seq_len is None:
            raise ValueError("seq_len must be provided")
        self.num_capsules = num_capsules
        self.routing_iters = routing_iters
        self.seq_len = seq_len
        # 实例化动态路由模块
        self.routing = Routing(seq_len, input_dim, num_capsules, routing_iters)
        # 两层全连接（MLP）对 capsule 表示进行进一步变换
        self.dense1 = nn.Linear(input_dim, 4 * input_dim)
        self.dense2 = nn.Linear(4 * input_dim, input_dim)

    def forward(self, sequence_emb, mask):
        """
        Args:
            sequence_emb: [batch_size, L, input_dim] 用户历史行为嵌入。
            mask: [batch_size, L] 布尔型张量，指示有效行为位置。
        Returns:
            capsules: [batch_size, num_capsules, input_dim] 多兴趣表示。
        """
        bs, L, D = sequence_emb.shape
        # 若 mask 长度与实际序列长度不一致，则做相应调整
        if mask.size(1) != L:
            if mask.size(1) < L:
                pad_size = L - mask.size(1)
                pad_mask = torch.zeros(bs, pad_size, dtype=mask.dtype, device=mask.device)
                mask = torch.cat([mask, pad_mask], dim=1)
            else:
                mask = mask[:, :L]
        # 根据 mask 计算每个样本的有效行为数，形状为 [batch_size, 1]
        seq_lens = mask.sum(dim=1, keepdim=True)
        # 调用动态路由模块得到 capsule 表示
        capsules = self.routing(sequence_emb, seq_lens)
        # 经过两层 MLP 进一步变换 capsule 表示
        capsules = self.dense2(F.relu(self.dense1(capsules)))
        return capsules

# 修改后的 MIND 模型，保持其他模块结构不变，仅更新多兴趣提取部分
class MIND(BaseModel):
    def __init__(self, feature_map, 
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 num_capsules=4,
                 routing_iters=3,
                 seq_len=65,  # 默认历史行为序列长度，根据实际情况调整
                 learning_rate=1e-3, 
                 embedding_dim=10, 
                 net_dropout=0, 
                 batch_norm=False, 
                 accumulation_steps=1,
                 embedding_regularizer=None, 
                 net_regularizer=None,
                 **kwargs):
        kwargs.pop("model_id", None)
        kwargs.pop("gpu", None)
        super(MIND, self).__init__(feature_map,
                                   model_id="MIND", 
                                   gpu=kwargs.get("gpu", -1), 
                                   embedding_regularizer=embedding_regularizer, 
                                   net_regularizer=net_regularizer,
                                   **kwargs)
        self.accumulation_steps = accumulation_steps
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        self.item_info_dim = 0
        # 统计所有物品相关特征的嵌入维度（只统计 item 特征）
        for feat, spec in self.feature_map.features.items():
            if spec.get("source") == "item":
                self.item_info_dim += spec.get("embedding_dim", embedding_dim)
        
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.multi_interest_extractor = MultiInterestExtractor(
            input_dim=self.item_info_dim,
            num_capsules=num_capsules,
            routing_iters=routing_iters,
            seq_len=seq_len
        )
        
        # 修改 _calculate_input_dim：不将 item 特征重复计入 non_meta_emb_dim
        input_dim = self._calculate_input_dim(feature_map, embedding_dim, num_capsules, self.item_info_dim)
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation, 
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def _calculate_input_dim(self, feature_map, embedding_dim, num_capsules, item_info_dim):
        """
        计算 DNN 的输入维度，只包含非物品特征和经过多兴趣提取后的 capsule 表示（扁平化后）。
        """
        non_meta_emb_dim = 0
        for feat, spec in feature_map.features.items():
            if feat in feature_map.labels:
                continue
            if spec.get("type") == "meta":
                continue
            # 排除物品特征，避免重复计入
            if spec.get("source") == "item":
                continue
            non_meta_emb_dim += spec.get("embedding_dim", embedding_dim)
        return non_meta_emb_dim + num_capsules * item_info_dim

    def get_inputs(self, inputs, feature_source=None):
        """
        根据 inputs 类型提取 batch_dict、item_dict 和 mask。
        """
        if isinstance(inputs, tuple):
            batch_dict, item_dict, mask = inputs
        elif isinstance(inputs, dict):
            batch_dict = inputs.get("batch_dict", {})
            item_dict = inputs.get("item_dict", {})
            mask = inputs.get("mask", None)
        else:
            raise ValueError("Inputs must be either a tuple or a dict.")
        
        X_dict = {}
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
        
        if mask is not None:
            mask = mask.to(self.device)
        
        return X_dict, item_dict, mask

    def get_group_id(self, inputs):
        """
        重写 get_group_id 方法，处理 tuple 格式的输入，
        假设 group_id 存储在 batch_dict 中，且对应键为 self.feature_map.group_id
        """
        if isinstance(inputs, tuple):
            batch_dict = inputs[0]
        elif isinstance(inputs, dict):
            batch_dict = inputs
        else:
            raise ValueError("Unsupported input type for get_group_id")
        return batch_dict[self.feature_map.group_id]

    def forward(self, inputs):
        batch_dict, item_dict, mask = self.get_inputs(inputs)
        emb_list = []
        
        # 用户侧普通特征嵌入
        if batch_dict:
            feature_emb = self.embedding_layer(batch_dict, flatten_emb=True)
            emb_list.append(feature_emb)
        
        # 物品或历史行为嵌入
        item_feat_emb = self.embedding_layer(item_dict, flatten_emb=True)
        batch_size = mask.shape[0]
        # 假设 item_feat_emb 原始形状为 (batch_size, seq_len * item_info_dim)，重塑为 (batch_size, seq_len, item_info_dim)
        item_feat_emb = item_feat_emb.view(batch_size, -1, self.item_info_dim)
        
        # 利用动态路由提取多兴趣表示
        multi_interest = self.multi_interest_extractor(item_feat_emb, mask)
        multi_interest_flat = multi_interest.view(batch_size, -1)
        emb_list.append(multi_interest_flat)
        
        # 拼接所有嵌入作为 DNN 的输入，并返回 {"y_pred": y_pred}
        feature_emb = torch.cat(emb_list, dim=-1)
        y_pred = self.dnn(feature_emb)
        return {"y_pred": y_pred}

    def get_labels(self, inputs):
        """
        重写 get_labels 方法，处理 tuple 格式的输入，并确保返回的标签形状与预测一致，
        且 dtype 为 Float。
        假设 inputs 为 (batch_dict, item_dict, mask)，标签存储在 batch_dict 中。
        """
        if isinstance(inputs, tuple):
            batch_dict = inputs[0]
            labels = self.feature_map.labels
            y = batch_dict[labels[0]].to(self.device)
        elif isinstance(inputs, dict):
            labels = self.feature_map.labels
            y = inputs[labels[0]].to(self.device)
        else:
            raise ValueError("Unsupported input type for get_labels")
        
        # 如果标签是一维的，则扩展最后一个维度
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        return y.float()

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
        self._batch_index += 1
        return loss
