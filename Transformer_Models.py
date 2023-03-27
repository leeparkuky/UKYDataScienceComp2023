import datasets
from transformers import AutoModel, AutoConfig
import sklearn
import torch.nn as nn
import os
from dataclasses import dataclass
from transformers.utils import ModelOutput
from typing import Optional, Tuple
from transformers import DistilBertModel
import torch


class ContextDistilBert(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.bert_title = DistilBertModel.from_pretrained(model_dir)
        self.bert_content = DistilBertModel.from_pretrained(model_dir)
    
    def forward(self, attention_mask_title = None, input_ids_title = None, 
                    attention_mask_content = None, input_ids_content = None, **kwargs):
        input_dict_title = {'input_ids': input_ids_title,
                     'attention_mask':attention_mask_title}
        input_dict_content = {'input_ids': input_ids_content,
                     'attention_mask':attention_mask_content}
        outputs_1 = self.bert_title(**input_dict_title)
        outputs_1 = outputs_1[0][:,0, :]
        outputs_2 = self.bert_content(**input_dict_content)
        outputs_2 = outputs_2[0][:,0, :]
        output = torch.cat([outputs_1, outputs_2], dim = -1)
        return output
        

class ContextDistilBertwithData(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.contextDistilBert = ContextDistilBert(model_dir)
        
    def forward(self, attention_mask_title = None, input_ids_title = None, 
                    attention_mask_content = None, input_ids_content = None, **kwargs):
        input_dict = {'input_ids_title': input_ids_title,
                     'attention_mask_title':attention_mask_title,
                     'input_ids_content': input_ids_content,
                     'attention_mask_content':attention_mask_content}
        x = self.contextDistilBert(**input_dict)
        fernandes_data = torch.cat([v.reshape(-1, 1) for k,v in kwargs.items() if 'shares' not in k], dim = -1)
        return torch.cat([x,fernandes_data], dim = -1)
        
        

# class ContextDistilBertForClassification(nn.Module):
#     def __init__(self, model_dir):
#         model_ckpt = model_dir
#         super().__init__()
#         self.bert_title = AutoModel.from_pretrained(model_ckpt) # body_1
#         self.bert_content = AutoModel.from_pretrained(model_ckpt)
#         self.config = AutoConfig.from_pretrained(model_ckpt) # config
#         self.linear_gelu_stack = nn.Sequential(
#             nn.Dropout(.1),
#             nn.Linear(self.config.dim * 2, self.config.hidden_dim),
#             nn.GELU(),
#             nn.Linear(self.config.hidden_dim, 1),
#         )
        
#     def forward(self, input_ids = None, attention_mask_title = None, input_ids_title = None, 
#                     attention_mask_content = None, input_ids_content = None,  shares = None, **kwargs):
#         input_dict_title = {'input_ids': input_ids_title,
#                      'attention_mask':attention_mask_title}
#         input_dict_content = {'input_ids': input_ids_content,
#                      'attention_mask':attention_mask_content}
#         outputs_1 = self.bert_title(**input_dict_title, **kwargs)
#         outputs_1 = outputs_1[0][:,0, :]
#         outputs_2 = self.bert_content(**input_dict_content, **kwargs)
#         outputs_2 = outputs_2[0][:,0, :]
#         output = torch.cat([outputs_1, outputs_2], dim = -1)
#         reg_output = self.linear_gelu_stack(output)
        
#         # calculate losses
#         loss = None
#         if shares is not None:
#             loss_fct = nn.PoissonNLLLoss()
#             loss = loss_fct(reg_output.view(-1), shares.view(-1))
#         # return model output object
#         return PoissonRegressionOutput(loss = loss, 
#                                       prediction = reg_output, 
#                                       hidden_states = output)









































