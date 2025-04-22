#utils/baseline.py
import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel, Data2VecAudioModel
from utils.cross_attn_encoder import CMELayer, BertConfig
import math

# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_generator = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, cls_tokens):
        context = torch.mean(cls_tokens, dim=1)  # [batch_size, 768]
        query = self.query_generator(context).unsqueeze(2)  # [batch_size, 768, 1]
        scores = torch.matmul(cls_tokens, query)  # [batch_size, 4, 1]
        weights = self.softmax(scores)  # [batch_size, 4, 1]
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
        self.roberta_model = RobertaModel.from_pretrained('roberta-large')
        self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-large-960h")

        # 文本和音频输出层
        self.T_output_layers = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dimension, 1)
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.dimension, 1)
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
            nn.Linear(self.config.dimension if self.use_attnFusion, 512),  # 输入维度改为 768
            nn.ReLU(),
            nn.Linear(512, 1)
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

    def forward(self, text_inputs, text_mask,audio_inputs, audio_mask,):
        # 文本特征提取
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        T_hidden_states = raw_output.last_hidden_state
        input_pooler = raw_output["pooler_output"]

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

        # 单模态输出（保持不变）
        T_features = torch.cat(input_pooler, dim=1)
        A_features_output = torch.cat(A_features, dim=1)
        T_output = self.T_output_layers(T_features)
        A_output = self.A_output_layers(A_features_output)

        text_inputs, text_attn_mask = self.prepend_cls(T_hidden_states, text_mask, 'text')  # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(A_hidden_states, audio_mask_new, 'audio')  # add cls token

        for layer_module in self.CME_layers:
            text_inputs, audio_inputs = layer_module(text_inputs, text_attn_mask,
                                                     audio_inputs, audio_attn_mask)

        text_cls = text_inputs[:, 0, :]
        audio_cls = audio_inputs[:, 0, :]

        if self.config.use_attnFusion:
            cls_tokens = torch.stack([text_cls, audio_cls, ], dim=1)  # [batch_size, 4, 768]
            fused_hidden_states = self.attention_fusion(cls_tokens)  # [batch_size, 768]
            fused_output = self.fused_output_layers(fused_hidden_states)
        else:
            fused_hidden_states = torch.cat((text_inputs[:, 0, :], audio_inputs[:, 0, :]), dim=1)  # Shape is [batch_size, 1024*4]

            fused_output = self.fused_output_layers(fused_hidden_states)  # Shape is [batch_size, 1]




        output = {
            'T': T_output,
            'A': A_output,
            'M': fused_output,
        }
        return output