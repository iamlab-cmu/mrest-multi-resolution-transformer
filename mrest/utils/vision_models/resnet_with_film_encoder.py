from typing import Any, Dict, List, Mapping, Optional, Union
import copy
import numpy as np
import re

import torch
import torch.nn as nn

import torchvision.models as models
import torchvision.transforms as T
from omegaconf import DictConfig

from mrest.utils.vision_models.resnet_base import resnet18, resnet34


class ResnetFilmImageEncoder(nn.Module):
    """
    Load pretrained or randomly initialized resnet models from torchvision.
    """
    def __init__(self, embedding_name: str, load_path: str, film_config: DictConfig,
                 *args, **kwargs) -> None:
        super().__init__()

        self.embedding_name = embedding_name
        self.append_object_mask = kwargs.get('append_object_mask', None)

        pretrained = load_path != "random"
        weights = 'IMAGENET1K_V1' if load_path != 'random' else None
        if embedding_name == 'resnet34_film':
            resnet_film_model = resnet34(weights=weights, film_config=film_config)
            embedding_dim = 512
        elif embedding_name == 'resnet18_film':
            resnet_film_model = resnet18(weights=weights, film_config=film_config)
            embedding_dim = 512
        elif embedding_name == 'resnet50_film':
            raise NotImplementedError
        else:
            print("Requested model not available currently")
            raise NotImplementedError

        # Copy over weights from model to film_model (all weights should correspond)
        self.embedding_dim = embedding_dim
        self.resnet_film_model = resnet_film_model
        
        # FiLM config
        self.film_config = film_config
        if film_config.use:
            film_models = []
            for layer_idx, num_blocks in enumerate(self.resnet_film_model.layers):
                if layer_idx in film_config.use_in_layers:
                    num_planes = self.resnet_film_model.film_planes[layer_idx]
                    film_model_layer = nn.Linear(
                        film_config.task_embedding_dim, num_blocks * 2 * num_planes)
                else:
                    film_model_layer = None
                film_models.append(film_model_layer)

            self.film_models = nn.ModuleList(film_models)

        # make FC layers to be identity
        # NOTE: This works for ResNet backbones but should check if same
        # template applies to other backbone architectures
        self.resnet_film_model.fc = nn.Identity()

        augment_img = kwargs['augment_img']
        initial_resize = kwargs['initial_resize']
        use_crop_aug_during_eval = kwargs['use_crop_aug_during_eval']
        resize_tfs = [
            T.Resize(initial_resize),
            T.CenterCrop(224),
        ]
        if self.append_object_mask is not None:
            normalize_tfs = [
                T.Normalize([0.485, 0.456, 0.406, 0.], [0.229, 0.224, 0.225, 1.])
            ]
        else:
            normalize_tfs = [
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]

        if augment_img:
            augment_padding = kwargs['augment_padding']
            shift_tfs = [ T.RandomCrop(224, padding=augment_padding, padding_mode='edge') ]
            transforms = resize_tfs + shift_tfs + normalize_tfs
        else:
            transforms = resize_tfs + normalize_tfs
        self.transforms = T.Compose(transforms)
        # Transforms for policy evaluation.
        if use_crop_aug_during_eval:
            self.eval_transforms = copy.deepcopy(self.transforms)
        else:
            self.eval_transforms = T.Compose(resize_tfs + normalize_tfs)

    def forward(self, inp,
                tasks: Optional[List[str]] = None,
                task_embs: Optional[torch.Tensor] = None,
                task_descriptions: Optional[List[str]] = None,
                obs_info: Optional[Dict[str, Any]] = None,
                camera_name=None) -> torch.Tensor:
        film_outputs = []
        for layer_idx, num_blocks in enumerate(self.resnet_film_model.layers):
            if self.film_config.use and self.film_models[layer_idx] is not None:
                film_features = self.film_models[layer_idx](task_embs)
            else:
                film_features = None
            film_outputs.append(film_features)
        return self.resnet_film_model(inp, film_outputs)

    @property
    def output_embedding_size(self) -> int:
        return self.embedding_dim

    def get_trainable_parameters(self):
        return self.parameters()

    def get_image_encoder_parameters(self, return_param_groups: bool = False):
        """Return image encoder parameters only. If the image encoder is frozen
        these parameters will not be updated."""
        if return_param_groups:
            return {
                'vision_backbone_params': self.resnet_film_model.parameters()
            }
        else:
            return self.resnet_film_model.parameters()
    
    def get_nonimage_encoder_parameters(self):
        """Return any non-image encoder parameters. 
        
        Even if the image encoder is frozen these parameters will still be updated.
        This is used for down_projection layers or task embeddings (e.g. FiLM modules).
        """
        if self.film_config.use:
            return self.film_models.parameters()
        else:
            return []
