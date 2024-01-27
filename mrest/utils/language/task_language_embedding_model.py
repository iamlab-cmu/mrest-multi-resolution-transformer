import numpy as np
import torch
import torch.nn as nn

from mrest.utils.language.language_models import *
from typing import List, Mapping


class TaskEmbeddingController(nn.Module):
    """Main module controlling task embeddings."""

    def __init__(self, config):
        super(TaskEmbeddingController, self).__init__()
        self.device = config.device

        self.cfg = config
        self.train_task_embeddings = config.train_task_embeddings

        self.task_embedding_dim = None
        if 'roberta' in config['language_model']:
            self.lang_model = RobertaSmall(config)
            self.task_embedding_dim = self.lang_model.output_embedding_size
        elif 'distilBERT' in config['language_model']:
            self.lang_model = DistilBERT(config)
            self.task_embedding_dim = self.lang_model.output_embedding_size
        elif 'random' in config['language_model']:
            self.task_embedding_dim = config.random.task_embedding_dim
        else:
            raise ValueError("Language model not implemented")
            self.task_embedding_dim = self.lang_model.output_embedding_size

        if self.train_task_embeddings:
            # Default in MetaAdapters
            # self.task_hyper_net = TaskHyperNet(config)
            raise NotImplementedError()

        self.task_to_embeddings = nn.ParameterDict(dict())
        self.task_to_descriptions = {}

    def get_trainable_parameters(self, return_param_groups: bool = False):
        if self.train_task_embeddings:
            raise NotImplementedError()
        return []
        
    def get_task_embedding_from_description(self, task_descriptions: List[str]):
        """Get task embedding from task description."""
        return self.lang_model(task_descriptions)
    
    def cache_task_embedding_for_tasks(self, task_desc_by_names: Mapping[str, List[str]]):
        for task_name, task_descriptions in task_desc_by_names.items():
            embs = self.get_task_embedding_from_description(task_descriptions)
            # assert self.task_to_embeddings.get(task_name) is None, (
            #     'Overwriting existing task embedding')
            self.task_to_embeddings[task_name] = embs
            self.task_to_descriptions[task_name] = task_descriptions

    def set_random_task_embeddings(self, tasks):
        self.task_to_embeddings = nn.ParameterDict(dict())
        for task in tasks:
            task_embedding = torch.Tensor(torch.randn(self.task_embedding_dim)).to(self.device)
            self.task_to_embeddings[task] = nn.Parameter(task_embedding)
    
    def set_lang_cond_task_embeddings(self, tasks):
        self.task_to_embeddings = self.lang_model(tasks)
        self.task_embedding_dim = self.lang_model.output_embedding_size

    def forward(self, task_names: List[str], randomize_task_emb_sampling: bool = False):
        """Returns cached task embedding."""
        task_embs = []
        for task_name in task_names:
            task_emb = self.task_to_embeddings[task_name]
            if task_emb.size(0) > 1 and randomize_task_emb_sampling:
                task_emb_idx = np.random.choice(task_emb.size(0))
                task_embs.append(task_emb[task_emb_idx:task_emb_idx + 1])
            else:
                task_embs.append(task_emb[0:1])
        return torch.cat(task_embs, dim=0)
