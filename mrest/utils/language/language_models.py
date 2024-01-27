import json
import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer


class RobertaSmall(nn.Module):
    def __init__(self, config):
        super(RobertaSmall, self).__init__()

        self.device = config.device
        with open(config.roberta.load_path) as f:
            self.metadata = json.load(f)
        
        self.task_embedding_dim = 768

    def forward(self, tasks):
        task_to_embeddings = nn.ParameterDict(dict())
        for task in tasks:
            task_embedding = torch.Tensor(self.metadata[task.split('-v2')[0] + '-v1']).to(self.device)
            task_to_embeddings[task] = nn.Parameter(task_embedding)
        return task_to_embeddings
    
    @property
    def output_embedding_size(self):
        return self.task_embedding_dim


class DistilBERT(nn.Module):
    def __init__(self, config):
        super(DistilBERT, self).__init__()

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.task_embedding_dim = 768
        self.max_len = config.distilBERT.token_max_len
        self.device = config.device
    
    def forward(self, tasks):
        ids, masks, token_type_ids = [], [], []
        for i, task in enumerate(tasks):
            inputs = self.tokenizer.encode_plus(task, return_tensors="pt", truncation=True, max_length=self.max_len, pad_to_max_length=True)
            ids.append(inputs['input_ids'])
            masks.append(inputs['attention_mask'])
        ids = torch.cat(ids, dim=0)
        masks = torch.cat(masks, dim=0)
        
        outputs = self.model(input_ids=ids, attention_mask=masks)
        last_hidden_states = outputs.last_hidden_state[:, 0, :]
        task_embeddings = torch.Tensor(last_hidden_states).to(self.device)
        return task_embeddings

    @property
    def output_embedding_size(self):
        return self.task_embedding_dim
