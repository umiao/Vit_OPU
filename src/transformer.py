"""
Adapted from https://github.com/lukemelas/simple-bert
"""
 
import numpy as np
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
import torch
from .pwlin import pwlin_gelu_basic
from .utils import float2fix, auto_quantize, sqrt_cordic, arctan_cordic

import src.config


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout, idx):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.layer_idx = idx
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        use_static_value, do_quantization = src.config.use_static_value, src.config.do_quantization
        quantize_param_dict = src.config.quantize_param_dict
        idx = self.layer_idx
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(idx) + 'attn_0']
            q = float2fix(q, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(q)
            q = float2fix(q, mantissa_bits)
            quantize_param_dict['transformer_' + str(idx) + 'attn_0'] = mantissa_bits

        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(idx) + 'attn_1']
            k = float2fix(k, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(k)
            k = float2fix(k, mantissa_bits)
            quantize_param_dict['transformer_' + str(idx) + 'attn_1'] = mantissa_bits

        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(idx) + 'attn_2']
            v = float2fix(v, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(v)
            v = float2fix(v, mantissa_bits)
            quantize_param_dict['transformer_' + str(idx) + 'attn_2'] = mantissa_bits

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])

        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        # np.sqrt(k.size(-1))
        base = sqrt_cordic(k.size(-1))
        scores = q @ k.transpose(-2, -1)
        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(idx) + 'attn_3']
            scores = float2fix(scores, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(scores)
            quantize_param_dict['transformer_' + str(idx) + 'attn_3'] = mantissa_bits
            scores = float2fix(scores, mantissa_bits)

        scores = scores / base
        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(idx) + 'attn_4']
            scores = float2fix(scores, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(scores)
            quantize_param_dict['transformer_' + str(idx) + 'attn_4'] = mantissa_bits
            scores = float2fix(scores, mantissa_bits)


        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)

        scores = self.drop(F.softmax(scores, dim=-1))
        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(idx) + 'attn_5']
            scores = float2fix(scores, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(scores)
            quantize_param_dict['transformer_' + str(idx) + 'attn_5'] = mantissa_bits
            scores = float2fix(scores, mantissa_bits)

        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(idx) + 'attn_6']
            h = float2fix(h, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(h)
            quantize_param_dict['transformer_' + str(idx) + 'attn_6'] = mantissa_bits
            h = float2fix(h, mantissa_bits)

        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim, idx):
        super().__init__()
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.layer_idx = idx

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        use_static_value, do_quantization = src.config.use_static_value, src.config.do_quantization
        quantize_param_dict = src.config.quantize_param_dict

        layer_idx = self.layer_idx
        x = self.fc1(x)

        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'Positional_0']
            x = float2fix(x, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(x)
            x = float2fix(x, mantissa_bits)
            quantize_param_dict['transformer_' + str(layer_idx) + 'Positional_0'] = mantissa_bits

        # x = F.gelu(x)
        res = F.gelu(x).cpu().numpy()



        x = x.cpu().numpy()
        mantissa_bits_in = quantize_param_dict['transformer_' + str(layer_idx) + 'Positional_0']
        mantissa_bits_out = quantize_param_dict['transformer_' + str(layer_idx) + 'Positional_1']
        x = pwlin_gelu_basic(x, None, mantissa_bits_in, mantissa_bits_out)
        # print(np.mean(np.abs(res - x)))
        src.config.accumulative_err += np.mean(np.abs(res - x))
        x = torch.from_numpy(x).cuda().float()



        # vfunc = np.vectorize(arctan_cordic)
        # # tmp_val = 1 + vfunc(np.sqrt(np.pi / 2),  (x + 0.044715 * (x ** 3)))   # np.sqrt(2 / np.pi) * (x + 0.044715 * (x ** 3))
        #
        # tmp_val = 1 + np.arctan((x + 0.044715 * (x ** 3)) / np.sqrt(np.pi / 2))
        #
        # # tmp_val = torch.from_numpy(np.array([tmp_val])).cuda()
        # # if use_static_value:
        # #     mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'gelu_0']
        # #     tmp_val = float2fix(tmp_val, mantissa_bits)
        # # else:
        # #     mantissa_bits = auto_quantize(tmp_val)
        # #     quantize_param_dict['transformer_' + str(layer_idx) + 'gelu_0'] = mantissa_bits
        # #     tmp_val = float2fix(tmp_val, mantissa_bits)
        # # tmp_val = tmp_val.cpu().numpy()
        # tmp_val = tmp_val * 0.5 * x
        # tmp_val = torch.from_numpy(np.array(tmp_val)).cuda()
        # # if use_static_value:
        # #     mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'gelu_1']
        # #     tmp_val = float2fix(tmp_val, mantissa_bits)
        # # else:
        # #     mantissa_bits = auto_quantize(tmp_val)
        # #     quantize_param_dict['transformer_' + str(layer_idx) + 'gelu_1'] = mantissa_bits
        # #     tmp_val = float2fix(tmp_val, mantissa_bits)
        # # x = tmp_val
        # err_res = tmp_val.cpu().numpy()
        # print('The mean relative error would be: ', np.mean(np.abs(res.cpu().numpy() - err_res) / res.cpu().numpy()))
        # x = tmp_val.float()

        # 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi)) ) * (x + 0.044715 * np.pow(x, 3))

        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'Positional_1']
            x = float2fix(x, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(x)
            quantize_param_dict['transformer_' + str(layer_idx) + 'Positional_1'] = mantissa_bits
            x = float2fix(x, mantissa_bits)

        x = self.fc2(x)
        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'Positional_2']
            x = float2fix(x, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(x)
            quantize_param_dict['transformer_' + str(layer_idx) + 'Positional_2'] = mantissa_bits
            x = float2fix(x, mantissa_bits)

        return x

############################



class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, idx=0):
        super().__init__()
        self.layer_idx = idx
        self.attn = MultiHeadedSelfAttention(dim, num_heads, dropout, idx)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionWiseFeedForward(dim, ff_dim, idx)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)


    def my_norm(self, x, idx):
        use_static_value, do_quantization = src.config.use_static_value, src.config.do_quantization
        quantize_param_dict = src.config.quantize_param_dict
        layer_idx = self.layer_idx
        dim = 2
        keepdim = True
        var_val, m_val = torch.var_mean(x, dim=dim, keepdim=keepdim, unbiased=False)
        # quantization
        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'norm_' + str(idx) + '_0']
            var_val = float2fix(var_val, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(var_val)
            quantize_param_dict['transformer_' + str(layer_idx) + 'norm_' + str(idx) + '_0'] = mantissa_bits
            var_val = float2fix(var_val, mantissa_bits)



        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'norm_' + str(idx) + '_1']
            m_val = float2fix(m_val, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(m_val)
            quantize_param_dict['transformer_' + str(layer_idx) + 'norm_' + str(idx) + '_1'] = mantissa_bits
            m_val = float2fix(m_val, mantissa_bits)

        std_val = torch.add(var_val, 1e-6)
        # std_val = torch.sqrt(std_val)
        std_val = std_val.cpu().numpy()
        a, b, c = std_val.shape
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    std_val[i, j, k] = sqrt_cordic(std_val[i, j, k])
        std_val = torch.from_numpy(std_val).cuda()

        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'norm_' + str(idx) + '_2']
            std_val = float2fix(std_val, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(std_val)
            quantize_param_dict['transformer_' + str(layer_idx) + 'norm_' + str(idx) + '_2'] = mantissa_bits
            std_val = float2fix(std_val, mantissa_bits)

        x = torch.sub(x, m_val)


        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'norm_' + str(idx) + '_3']
            x = float2fix(x, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(x)
            quantize_param_dict['transformer_' + str(layer_idx) + 'norm_' + str(idx) + '_3'] = mantissa_bits
            x = float2fix(x, mantissa_bits)

        x = torch.div(x, std_val)
        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'norm_' + str(idx) + '_4']
            x = float2fix(x, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(x)
            quantize_param_dict['transformer_' + str(layer_idx) + 'norm_' + str(idx) + '_4'] = mantissa_bits
            x = float2fix(x, mantissa_bits)

        if idx == 1:
            weight = self.norm1.weight
            bias = self.norm1.bias
        else:
            weight = self.norm2.weight
            bias = self.norm2.bias

        x = torch.mul(x, weight)
        x = torch.add(x, bias)

        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'norm_' + str(idx) + '_5']
            x = float2fix(x, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(x)
            quantize_param_dict['transformer_' + str(layer_idx) + 'norm_' + str(idx) + '_5'] = mantissa_bits
            x = float2fix(x, mantissa_bits)

        return x

    def forward(self, x, mask):
        use_static_value, do_quantization = src.config.use_static_value, src.config.do_quantization
        quantize_param_dict = src.config.quantize_param_dict
        layer_idx = self.layer_idx
        original_x = x
        # x = self.norm1(x)
        x = self.my_norm(x, 1)

        x = self.attn(x, mask)

        x = self.proj(x)


        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'Proj_0']
            x = float2fix(x, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(x)
            quantize_param_dict['transformer_' + str(layer_idx) + 'Proj_0'] = mantissa_bits
            x = float2fix(x, mantissa_bits)

        h = self.drop(x)

        x = original_x + h

        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'add_0']
            x = float2fix(x, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(x)
            quantize_param_dict['transformer_' + str(layer_idx) + 'add_0'] = mantissa_bits
            x = float2fix(x, mantissa_bits)

        norm2 = self.my_norm(x, 2)
        # norm2 = self.norm2(x)
        pwff = self.pwff(norm2)

        h = self.drop(pwff)
        x = x + h


        if use_static_value and do_quantization:
            mantissa_bits = quantize_param_dict['transformer_' + str(layer_idx) + 'add_1']
            x = float2fix(x, mantissa_bits)
        elif do_quantization:
            mantissa_bits = auto_quantize(x)
            quantize_param_dict['transformer_' + str(layer_idx) + 'add_1'] = mantissa_bits
            x = float2fix(x, mantissa_bits)
        return x


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, ff_dim, dropout, i) for i in range(num_layers)])

        self.intermediate_res_lis = []

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
