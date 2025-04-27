# utils/model.py
import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel, Data2VecAudioModel
from utils.cross_attn_encoder import CMELayer, BertConfig
import math

# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, hidden_dim, nheads):
        """
        初始化多头注意力融合模块。

        参数：
            hidden_dim (int): 隐藏层维度（例如 768）。
            nheads (int): 注意力头的数量。
        """
        super().__init__()
        # 初始化多头注意力模块
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nheads,
            dropout=0.0  # 与原始代码保持一致，不使用 dropout
        )

    def forward(self, cls_tokens):
        """
        前向传播，融合 CLS 标记。

        参数：
            cls_tokens (torch.Tensor): 输入的 CLS 标记，形状为 [batch_size, 4, hidden_dim]

        返回：
            torch.Tensor: 融合后的表示，形状为 [batch_size, hidden_dim]
        """
        # 计算全局上下文（CLS 标记的平均值）
        context = torch.mean(cls_tokens, dim=1)  # [batch_size, hidden_dim]

        # 准备多头注意力的输入
        query = context.unsqueeze(0)  # [1, batch_size, hidden_dim]
        key = cls_tokens.permute(1, 0, 2)  # [4, batch_size, hidden_dim]
        value = key  # 键和值相同，[4, batch_size, hidden_dim]

        # 计算多头注意力
        attn_output, _ = self.multihead_attn(query, key, value)  # attn_output: [1, batch_size, hidden_dim]
        fused = attn_output.squeeze(0)  # [batch_size, hidden_dim]

        # 添加残差连接
        residual = context  # [batch_size, hidden_dim]
        fused_with_residual = fused + residual  # [batch_size, hidden_dim]

        return fused_with_residual

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 添加一个线性层，用于从CLS标记的平均值生成动态查询
        self.query_generator = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, cls_tokens):
        # 输入：cls_tokens [batch_size, 4, 768]

        # 计算全局上下文（CLS标记的平均值）
        context = torch.mean(cls_tokens, dim=1)  # [batch_size, 768]

        # 动态生成查询向量
        query = self.query_generator(context).unsqueeze(2)  # [batch_size, 768, 1]

        # 计算注意力分数
        scores = torch.matmul(cls_tokens, query)  # [batch_size, 4, 1]
        weights = self.softmax(scores)  # [batch_size, 4, 1]

        # 加权融合
        fused = torch.sum(weights * cls_tokens, dim=1)  # [batch_size, 768]

        # 添加残差连接
        residual = torch.mean(cls_tokens, dim=1)  # [batch_size, 768]
        fused_with_residual = fused + residual  # [batch_size, 768]

        return fused_with_residual

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 预训练模型加载
        self.roberta_model = RobertaModel.from_pretrained('siebert/sentiment-roberta-large-english')
        self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-large-960h")
        # 文本和音频输出层
        self.T_output_layers = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dimension * 2, 1)
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dimension * 2, 1)
        )

        # CLS 嵌入
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=self.config.dimension)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=self.config.dimension)

        # CME 层配置
        Bert_config = BertConfig(
            num_hidden_layers=self.config.num_hidden_layers,
            hidden_size=self.config.dimension,
            intermediate_size=self.config.dimension * 4,
            num_attention_heads=self.config.nheads
        )
        self.CME_layers = nn.ModuleList([CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)])
        self.attention_fusion = AttentionFusion(self.config.dimension)

        # 多模态融合输出层
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.config.dimension if self.config.use_attnFusion else self.config.dimension*4, 768),  # 输入维度改为 768
            nn.ReLU(),
            nn.Linear(768, 1)
        )

        # 注意力投影层，用于计算上下文相关性
        self.text_context_attn = nn.Linear(self.config.dimension, self.config.dimension)
        self.audio_context_attn = nn.Linear(self.config.dimension, self.config.dimension)

        # 可调参数 alpha，控制时序偏置的强度
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始值为 0.5，可根据实验调整

    def prepend_cls(self, inputs, masks, layer_name):
        """添加 CLS 标记"""
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)
        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks.bool()), dim=1)
        return outputs, masks

    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask,
                audio_inputs, audio_mask, audio_context_inputs, audio_context_mask):
        # 文本特征提取
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        T_hidden_states = raw_output.last_hidden_state
        input_pooler = raw_output["pooler_output"]

        raw_output_context = self.roberta_model(text_context_inputs, text_context_mask, return_dict=True)
        T_context_hidden_states = raw_output_context.last_hidden_state
        context_pooler = raw_output_context["pooler_output"]

        # 音频特征提取（保留原始池化方案）
        audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        A_features = []
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(audio_out.attentions[layer][batch][0][0] != 0)
                    audio_mask_idx_new.append(padding_idx)
                    truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx], 0)
                    A_features.append(truncated_feature)
                    break
                except:
                    layer += 1
        A_features = torch.stack(A_features, 0).to(device)
        audio_mask_new = torch.zeros(A_hidden_states.shape[0], A_hidden_states.shape[1]).to(device)
        for batch in range(audio_mask_new.shape[0]):
            audio_mask_new[batch][:audio_mask_idx_new[batch]] = 1

        audio_context_out = self.data2vec_model(audio_context_inputs, audio_context_mask, output_attentions=True)
        A_context_hidden_states = audio_context_out.last_hidden_state
        A_context_features = []
        audio_context_mask_idx_new = []
        for batch in range(A_context_hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(audio_context_out.attentions[layer][batch][0][0] != 0)
                    audio_context_mask_idx_new.append(padding_idx)
                    truncated_feature = torch.mean(A_context_hidden_states[batch][:padding_idx], 0)
                    A_context_features.append(truncated_feature)
                    break
                except:
                    layer += 1
        A_context_features = torch.stack(A_context_features, 0).to(device)
        audio_context_mask_new = torch.zeros(A_context_hidden_states.shape[0],
                                            A_context_hidden_states.shape[1]).to(device)
        for batch in range(audio_context_mask_new.shape[0]):
            audio_context_mask_new[batch][:audio_context_mask_idx_new[batch]] = 1

        # 文本上下文注意力机制 + 时序偏置
        query_text = self.text_context_attn(input_pooler).unsqueeze(1)  # [batch_size, 1, hidden_size]
        key_text = T_context_hidden_states  # [batch_size, seq_len, hidden_size]
        value_text = T_context_hidden_states  # [batch_size, seq_len, hidden_size]
        attn_scores_text = torch.bmm(query_text, key_text.transpose(1, 2))  # [batch_size, 1, seq_len]

        # 添加时序偏置
        seq_len = T_context_hidden_states.size(1)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(attn_scores_text.size(0), seq_len)
        temporal_bias_text = self.alpha * torch.exp(-positions / 10.0).unsqueeze(1)  # [batch_size, 1, seq_len]
        attn_scores_text = attn_scores_text + temporal_bias_text

        attn_weights_text = F.softmax(
            attn_scores_text.masked_fill(text_context_mask.unsqueeze(1) == 0, float('-inf')), dim=-1)
        weighted_text_context = torch.bmm(attn_weights_text, value_text).squeeze(1)  # [batch_size, hidden_size]

        # 音频上下文注意力机制 + 时序偏置
        query_audio = self.audio_context_attn(A_features).unsqueeze(1)  # [batch_size, 1, hidden_size]
        key_audio = A_context_hidden_states  # [batch_size, audio_seq_len, hidden_size]
        value_audio = A_context_hidden_states  # [batch_size, audio_seq_len, hidden_size]
        attn_scores_audio = torch.bmm(query_audio, key_audio.transpose(1, 2))  # [batch_size, 1, audio_seq_len]

        # 添加时序偏置
        audio_seq_len = A_context_hidden_states.size(1)
        positions_audio = torch.arange(audio_seq_len, device=device).unsqueeze(0).expand(attn_scores_audio.size(0),
                                                                                         audio_seq_len)
        temporal_bias_audio = self.alpha * torch.exp(-positions_audio / 10.0).unsqueeze(
            1)  # [batch_size, 1, audio_seq_len]
        attn_scores_audio = attn_scores_audio + temporal_bias_audio

        attn_weights_audio = F.softmax(
            attn_scores_audio.masked_fill(audio_context_mask_new.unsqueeze(1) == 0, float('-inf')), dim=-1)
        weighted_audio_context = torch.bmm(attn_weights_audio, value_audio).squeeze(1)  # [batch_size, hidden_size]

        # 单模态输出（保持不变）
        T_features = torch.cat((input_pooler, context_pooler), dim=1)
        A_features_output = torch.cat((A_features, A_context_features), dim=1)
        T_output = self.T_output_layers(T_features)
        A_output = self.A_output_layers(A_features_output)

        text_inputs, text_attn_mask = self.prepend_cls(T_hidden_states, text_mask, 'text')  # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(A_hidden_states, audio_mask_new, 'audio')  # add cls token
        text_context_inputs = weighted_text_context.unsqueeze(1)  # [batch_size, 1, hidden_size]
        audio_context_inputs = weighted_audio_context.unsqueeze(1)  # [batch_size, 1, hidden_size]
        text_context_attn_mask = torch.ones(text_context_inputs.size(0), 1).to(device)
        audio_context_attn_mask = torch.ones(audio_context_inputs.size(0), 1).to(device)

        for layer_module in self.CME_layers:
            text_inputs, audio_inputs = layer_module(text_inputs, text_attn_mask,
                                                     audio_inputs, audio_attn_mask)

        for layer_module in self.CME_layers:
            text_context_inputs, audio_context_inputs = layer_module(text_context_inputs, text_context_attn_mask,
                                                                     audio_context_inputs, audio_context_attn_mask)

        text_cls = text_inputs[:, 0, :]
        audio_cls = audio_inputs[:, 0, :]
        text_context_cls = text_context_inputs[:, 0, :]
        audio_context_cls = audio_context_inputs[:, 0, :]

        if self.config.use_attnFusion:
            cls_tokens = torch.stack([text_cls, text_context_cls, audio_cls, audio_context_cls],dim=1)  # [batch_size, 4, 768]
            fused_hidden_states = self.attention_fusion(cls_tokens)  # [batch_size, 768]
            fused_output = self.fused_output_layers(fused_hidden_states)
        else:
            fused_hidden_states = torch.cat((text_inputs[:, 0, :], text_context_inputs[:, 0, :], audio_inputs[:, 0, :],
                                             audio_context_inputs[:, 0, :]), dim=1)  # Shape is [batch_size, 1024*4]

            fused_output = self.fused_output_layers(fused_hidden_states)  # Shape is [batch_size, 1]

        output = {
            'T': T_output,
            'A': A_output,
            'M': fused_output,
        }
        return output
