# utils/cross_attn_encoder.py
import torch
from torch import nn
import math

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=3,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="relu",
                 hidden_dropout_prob=0.3,
                 attention_probs_dropout_prob=0.3,
                 max_position_embeddings=512,
                 add_abs_pos_emb = False,
                 add_pos_enc = False):
        """Constructs BertConfig.
        Args:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            add_abs_pos_emb: absolute positional embeddings
            add_pos_enc: positional encoding
        """

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.add_abs_pos_emb = add_abs_pos_emb
        self.add_pos_enc = add_pos_enc

BertLayerNorm = torch.nn.LayerNorm

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.add_abs_pos_emb = config.add_abs_pos_emb
        if self.add_abs_pos_emb:
            self.abs_pos_emb = nn.Parameter(torch.randn(512, self.attention_head_size))
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)#(b, s,h, d) -> (b, h, s, d)

    def forward(self, hidden_states, context, attention_mask=None):
        #print(context.size(),attention_mask.size())
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        if self.add_abs_pos_emb:
            pos_emb = self.abs_pos_emb[0:context.size(1),:]
            pos_emb_q = self.abs_pos_emb[0:hidden_states.size(1),:]
            pos_emb_q = pos_emb_q.expand(query_layer.size(0), query_layer.size(1), -1, -1)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # shape is (b, h, s_q, s_k)
        if self.add_abs_pos_emb:
            attention_pos_scores = torch.matmul(query_layer+pos_emb_q, pos_emb.transpose(-1, -2))
            attention_scores = (attention_scores+attention_pos_scores) / math.sqrt(self.attention_head_size)
        else:
            attention_scores = attention_scores/ math.sqrt(self.attention_head_size)
            
        # Apply the attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand((-1,attention_scores.size(1),attention_scores.size(2),-1))
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
            #print(attention_mask.size())
            #print(attention_scores.size())
            attention_scores = attention_scores + attention_mask
            
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # shape is (b, h, s_q, d)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # shape is (b, s_q, h, d)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor) # attention_output = self.output(output, input_tensor)

        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


"""
---------------------------------------------------------------------------------------
      Above modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""


# class CMELayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # The cross-attention Layer
#         self.audio_attention = BertCrossattLayer(config)
#         self.lang_attention = BertCrossattLayer(config)
#
#         # Self-attention Layers
#         self.lang_self_att = BertSelfattLayer(config)
#         self.audio_self_att = BertSelfattLayer(config)
#
#         # Intermediate and Output Layers (FFNs)
#         self.lang_inter = BertIntermediate(config)
#         self.lang_output = BertOutput(config)
#         self.audio_inter = BertIntermediate(config)
#         self.audio_output = BertOutput(config)
#
#     def cross_att(self, lang_input, lang_attention_mask, audio_input, audio_attention_mask):
#         # Cross Attention
#         lang_att_output = self.lang_attention(lang_input, audio_input, ctx_att_mask=audio_attention_mask)
#         audio_att_output = self.audio_attention(audio_input, lang_input, ctx_att_mask=lang_attention_mask)
#         return lang_att_output, audio_att_output
#
#     def self_att(self, lang_input, lang_attention_mask, audio_input, audio_attention_mask):
#         # Self Attention
#         lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
#         audio_att_output = self.audio_self_att(audio_input, audio_attention_mask)
#         return lang_att_output, audio_att_output
#
#     def output_fc(self, lang_input, audio_input):
#         # FC layers
#         lang_inter_output = self.lang_inter(lang_input)
#         audio_inter_output = self.audio_inter(audio_input)
#
#         # Layer output
#         lang_output = self.lang_output(lang_inter_output, lang_input)
#         audio_output = self.audio_output(audio_inter_output, audio_input)
#         return lang_output, audio_output
#
#     def forward(self, lang_feats, lang_attention_mask,
#                       audio_feats, audio_attention_mask):
#
#         lang_att_output = lang_feats
#         audio_att_output = audio_feats
#
#         lang_att_output, audio_att_output = self.cross_att(lang_att_output, lang_attention_mask,
#                                                           audio_att_output, audio_attention_mask)
#         lang_att_output, audio_att_output = self.self_att(lang_att_output, lang_attention_mask,
#                                                          audio_att_output, audio_attention_mask)
#         lang_output, audio_output = self.output_fc(lang_att_output, audio_att_output)
#
#         return lang_output, audio_output


import torch.nn.functional as F

class CMELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化与原始 CMELayer 一致的子模块
        self.audio_attention = BertCrossattLayer(config)
        self.lang_attention = BertCrossattLayer(config)
        self.lang_self_att = BertSelfattLayer(config)
        self.audio_self_att = BertSelfattLayer(config)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.audio_inter = BertIntermediate(config)
        self.audio_output = BertOutput(config)

        # 协同注意力新增的可学习参数，用于融合注意力图
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))  # 初始化为 0.5，表示语言和音频初始贡献相等

    def compute_attention_scores(self, query, key, mask=None):
        """计算注意力得分并处理掩码
        Args:
            query: [batch_size, seq_len_q, hidden_size] - 查询向量
            key: [batch_size, seq_len_k, hidden_size] - 键向量
            mask: [batch_size, seq_len_k] - 注意力掩码
        Returns:
            attention_probs: [batch_size, seq_len_q, seq_len_k] - 归一化的注意力概率
        """
        # 计算原始注意力得分
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, seq_len_q, seq_len_k]

        # 处理掩码，确保不关注填充部分
        if mask is not None:
            # 扩展掩码维度以匹配注意力得分
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len_k]
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 归一化得到注意力概率
        attention_probs = F.softmax(attention_scores, dim=-1)
        return attention_probs

    def co_attention(self, lang_input, lang_attention_mask, audio_input, audio_attention_mask):
        """实现协同注意力机制
        Args:
            lang_input: [batch_size, lang_seq_len, hidden_size] - 语言特征
            lang_attention_mask: [batch_size, lang_seq_len] - 语言掩码
            audio_input: [batch_size, audio_seq_len, hidden_size] - 音频特征
            audio_attention_mask: [batch_size, audio_seq_len] - 音频掩码
        Returns:
            lang_output: [batch_size, lang_seq_len, hidden_size] - 更新后的语言特征
            audio_output: [batch_size, audio_seq_len, hidden_size] - 更新后的音频特征
        """
        # 获取维度信息
        batch_size, lang_seq_len, hidden_size = lang_input.size()
        audio_seq_len = audio_input.size(1)

        # 计算语言对音频的注意力
        lang_to_audio_attn = self.compute_attention_scores(lang_input, audio_input, audio_attention_mask)
        # [batch_size, lang_seq_len, audio_seq_len]

        # 计算音频对语言的注意力
        audio_to_lang_attn = self.compute_attention_scores(audio_input, lang_input, lang_attention_mask)
        # [batch_size, audio_seq_len, lang_seq_len]

        # 融合注意力图：使用可学习权重替代简单平均
        joint_attn = self.fusion_weight * lang_to_audio_attn + (1 - self.fusion_weight) * audio_to_lang_attn.transpose(1, 2)
        # [batch_size, lang_seq_len, audio_seq_len]

        # 使用联合注意力图生成加权特征
        lang_weighted = torch.bmm(joint_attn, audio_input)  # [batch_size, lang_seq_len, hidden_size]
        audio_weighted = torch.bmm(joint_attn.transpose(1, 2), lang_input)  # [batch_size, audio_seq_len, hidden_size]

        # 残差连接：结合原始特征，保持信息完整性
        lang_output = lang_input + lang_weighted
        audio_output = audio_input + audio_weighted

        return lang_output, audio_output

    def self_att(self, lang_input, lang_attention_mask, audio_input, audio_attention_mask):
        """自注意力层，与原始实现一致"""
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        audio_att_output = self.audio_self_att(audio_input, audio_attention_mask)
        return lang_att_output, audio_att_output

    def output_fc(self, lang_input, audio_input):
        """全连接层输出，与原始实现一致"""
        lang_inter_output = self.lang_inter(lang_input)
        audio_inter_output = self.audio_inter(audio_input)
        lang_output = self.lang_output(lang_inter_output, lang_input)
        audio_output = self.audio_output(audio_inter_output, audio_input)
        return lang_output, audio_output

    covariance = []
    def forward(self, lang_feats, lang_attention_mask, audio_feats, audio_attention_mask):
        """前向传播，使用协同注意力替代交叉注意力"""
        lang_att_output = lang_feats
        audio_att_output = audio_feats

        # 使用协同注意力处理模态交互
        lang_att_output, audio_att_output = self.co_attention(lang_att_output, lang_attention_mask,
                                                              audio_att_output, audio_attention_mask)

        # 自注意力进一步优化特征
        lang_att_output, audio_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                          audio_att_output, audio_attention_mask)

        # 全连接层生成最终输出
        lang_output, audio_output = self.output_fc(lang_att_output, audio_att_output)

        return lang_output, audio_output


