'''
copied from https://github.com/tjvsonsbeek/open-ended-medical-vqa/tree/main
'''

import numpy as np
from tqdm import tqdm
import sys
import os
import pdb
from typing import Tuple, Optional, Union

from peft import LoraConfig, get_peft_model,get_peft_config,PeftModelForCausalLM,TaskType,PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig 

import torch
import torch.nn as nn
from torch.nn import functional as nnf

import transformers
from transformers import set_seed, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from transformers.models.biogpt import BioGptForCausalLM, BioGptTokenizer, BioGptConfig
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(nn.Dropout(p=0.5))
                layers.append(act())
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

class VQAmedModel(nn.Module):
    def __init__(
        self,
        prefix_length=2,
        clip_length=2,
        prefix_size=512,
        num_layers=8,
        setting="lora",
        mapping_type="MLP",
        model_type="gpt2-xl",
    ):
        super(VQAmedModel, self).__init__()
        gpttype = model_type
        self.gpttype = gpttype
        self.setting = setting
        self.prefix_length = prefix_length
        # pdb.set_trace()
        self.gpt = AutoModelForCausalLM.from_pretrained(gpttype) # AutoModelForCausalLM.from_pretrained(gpttype,load_in_8bit=True,device_map='auto') if self.gpttype=="gpt2-xl" else AutoModelForCausalLM.from_pretrained(gpttype)
        # load the relevant fine-tuning strategy 
        if setting == "lora":
            # pdb.set_trace()
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1) if self.gpttype == "gpt2-xl" or self.gpttype == "stanford-crfm/BioMedLM"  \
                 else LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]) 
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prefixtuning":
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="p_tuning":
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prompttuning":
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=='frozen':
            for param in self.gpt.transformer.parameters():
                param.requires_grad = False
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpttype) # if gpttype == "gpt2-xl" else BioGptTokenizer.from_pretrained(gpttype)
        # pdb.set_trace()
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1] if self.gpttype == "gpt2-xl" else self.gpt.biogpt.layers[0].self_attn.k_proj.base_layer.in_features
        if mapping_type == "MLP":
            self.clip_project = MLP((
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                    self.gpt_embedding_size * prefix_length))
        elif mapping_type == "Transformer":
            self.clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                prefix_length,
                clip_length,
                num_layers)
        else:
            raise ValueError("select valid mapping type: MLP or Transformer")


    def forward(self, prefix, tokens, mask, q_len, batch_size):
        # pdb.set_trace()
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        
        if self.gpttype=='microsoft/biogpt':
            embedding = self.gpt.biogpt.embed_tokens(tokens)
        else:
            embedding = self.gpt.transformer.wte(tokens)  # tokens include "question: ", question, "context: ", img_embeding, "answer: ", answer, <endoftext>
        for b in range(batch_size):
            # insert the visual prefix after the question 
            # pdb.set_trace()
            embedding[b,q_len:q_len+self.prefix_length,:] = prefix_projections[b]  

        # pdb.set_trace()
        return self.gpt(inputs_embeds=embedding, attention_mask=mask)
    def generate(self, prefix, tokens, mask, q_len):
        # pdb.set_trace()
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        if self.gpttype=='microsoft/biogpt':
            # pdb.set_trace()
            embedding_txt = self.gpt.biogpt.embed_tokens(tokens)
        else:
            embedding_txt = self.gpt.transformer.wte(tokens)
        
        for b in range(prefix.shape[0]):
            # insert the visual prefix after the question 
            # pdb.set_trace()
            embedding_txt[b,q_len:q_len+self.prefix_length,:] = prefix_projections[b]  
        # pdb.set_trace()
        # embedding_txt[q_len:q_len+self.prefix_length,:] = prefix_projections
        return embedding_txt
    

# # adaptation of VQAmedModel for ablation studies
# class VQAmedModel_abl(nn.Module):
#     def forward(self, prefix, labels, tokens, mask, q_len, batch_size,abl):
#         embeddings = self.gpt.transformer.wte(tokens)
#         if abl=="replace_visual":
#             for b in range(batch_size):
#                 embeddings[b,q_len[b]:q_len[b]+self.prefix_length,:] = self.nv_tokens[b]  
#         elif abl=="remove_question":
#             prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
#             embeddings[:,q_len[0]:q_len[0]+self.prefix_length,:] = prefix_projections
#         elif abl=="swap":
#             prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
#             embeddings[:,q_len[0]:q_len[0]+self.prefix_length,:] = prefix_projections
#         return self.gpt(inputs_embeds=embeddings, attention_mask=mask)

#     def generate(self, prefix, labels, tokens, mask, q_len,abl):
#         prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
#         embeddings = self.gpt.transformer.wte(tokens)
#         if abl=="replace_visual":
#             embeddings[q_len:q_len+self.prefix_length,:] = self.nv_tokens[0]  
#         elif abl=="remove_question":
#             prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
#             embeddings[q_len:q_len+self.prefix_length,:] = prefix_projections
#         elif abl=="swap":
#             prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
#             embeddings[q_len:q_len+self.prefix_length,:] = prefix_projections
#         return embeddings

#     def __init__(
#         self,
#         prefix_length=2,
#         clip_length=2,
#         prefix_size=512,
#         num_layers=8,
#         setting="frozen",
#         mapping_type="MLP",
#         batch_size=32,
#     ):
#         super(VQAmedModel_abl, self).__init__()
#         gpttype = "gpt2-xl"
#         self.model_type = gpttype
#         self.setting = setting
#         self.prefix_length = prefix_length
#         self.gpt = GPT2LMHeadModel.from_pretrained(gpttype,load_in_8bit=True,device_map='auto')
#         if setting == "lora":
#             peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
#             self.gpt = get_peft_model(self.gpt,peft_config)
#         elif setting=="prefixtuning":
#             peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
#             self.gpt = get_peft_model(self.gpt,peft_config)
#         elif setting=="p_tuning":
#             peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
#             self.gpt = get_peft_model(self.gpt,peft_config)
#         elif setting=="prompttuning":
#             peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
#             self.gpt = get_peft_model(self.gpt,peft_config)
#         elif setting=='frozen':
#             for param in self.gpt.transformer.parameters():
#                 param.requires_grad = False
#         self.tokenizer = GPT2Tokenizer.from_pretrained(gpttype)
#         self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
#         # for the replace_visual ablation study we replace the visual tokens with learnable parameters 
#         self.nv_tokens = torch.nn.Parameter(torch.randn(batch_size,prefix_length,self.gpt_embedding_size),requires_grad=True).cuda()
#         if mapping_type == "MLP":
#             self.clip_project = MLP((prefix_size,
#                     (self.gpt_embedding_size * prefix_length) // 2,
#                     self.gpt_embedding_size * prefix_length,
#                     self.gpt_embedding_size * prefix_length))
#         elif mapping_type == "Transformer":
#             self.clip_project = TransformerMapper(
#                 prefix_size,
#                 self.gpt_embedding_size,
#                 prefix_length,
#                 clip_length,
#                 num_layers)
#         else:
#             raise ValueError("select valid mapping type: MLP or Transformer")
        