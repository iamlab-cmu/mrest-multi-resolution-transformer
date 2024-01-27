import copy
import itertools
import numpy as np
import pprint

from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from einops import reduce

from PIL import Image
import torchvision.models as models
import torchvision.transforms as T

from mrest.utils.mdetr.mdetr_model import MDETR
from mrest.utils.mdetr.backbone import build_backbone
from mrest.utils.mdetr.transformer import build_transformer, TransformerEncoderLayer, TransformerEncoder
from mrest.utils.mdetr.model_hub import mdetr_resnet101, mdetr_efficientnetB3, mdetr_efficientnetB5, mdetr_clevr

from mrest.utils.mdetr.position_encoding import build_position_encoding


class MDETRMultiViewBaselinePolicy(nn.Module):

    def __init__(self, embedding_name: str, load_path: str, mdetr_config: DictConfig,
                 *args, **kwargs):
        super().__init__()

        self.embedding_name = embedding_name
        self.mdetr_config = mdetr_config

        pretrained = load_path != "random"

        if mdetr_config.get('transformer_adapters'):
            if mdetr_config['transformer_adapters']['use']:
                assert mdetr_config.dropout == 0.0, 'Disable dropout for MDETR adapters'

        backbone = build_backbone(mdetr_config)
        params_require_grad = [name for name, parameter in backbone.named_parameters()
                               if parameter.requires_grad]
        print('==== Backbone Params that require grads ====')
        pprint.pprint(params_require_grad)
        transformer = build_transformer(mdetr_config)

        # Hard-coded from MDETR (how do we get 255)
        num_classes = 255
        qa_dataset = None

        use_all_encoded_img_tokens = True
        if mdetr_config.multiview.attn_type == 'only_static':
            use_all_encoded_img_tokens = False

        self.model = MDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=mdetr_config.num_queries,
            aux_loss=mdetr_config.aux_loss,
            contrastive_hdim=mdetr_config.contrastive_loss_hdim,
            contrastive_loss=mdetr_config.contrastive_loss,
            contrastive_align_loss=mdetr_config.contrastive_align_loss,
            qa_dataset=qa_dataset,
            split_qa_heads=mdetr_config.split_qa_heads,
            predict_final=mdetr_config.predict_final,
            use_control_readout_emb=mdetr_config.use_control_readout_emb,
            use_all_encoded_img_tokens=use_all_encoded_img_tokens,
        )

        self.attn_type = mdetr_config.multiview.attn_type
        assert self.attn_type in ('concat', 'only_static', 'only_hand')

        if self.attn_type == 'only_static' or self.attn_type == 'only_hand':
            # Use only static camera
            assert mdetr_config.use_control_readout_emb, 'Should set control embedding to True'
            assert mdetr_config.enc_layers == 6, 'Use same encoders as before (for reproducibility)'
            self.output_size = 256

        elif self.attn_type == 'concat':
            self.output_size = 512

            # Create Hand Camera model
            film_config = mdetr_config.multiview.concat.hand_cam_resnet_film_config
            self.create_model_for_hand_camera_image(film_config)

        else:
            raise ValueError('Not Implemented')

        criterion = None
        # NOTE: MDETR sets up different optimizers for different parts of the model
        # such as backbone, text_encoder etc.

        # TODO: We should create different models based on load_path
        if pretrained:
            self.use_pretrained_transformer = True
            if 'resnet101_no_transformer' in load_path:
                # Do not load pre-training weights for the transformer.
                self.use_pretrained_transformer = False
                pretrained_model = mdetr_resnet101(pretrained=pretrained, return_postprocessor=False)
            elif 'resnet101' in load_path:
                assert mdetr_config['backbone'] == 'resnet101'
                pretrained_model = mdetr_resnet101(pretrained=pretrained, return_postprocessor=False)
            elif 'efficientnetB3' in load_path:
                assert mdetr_config['backbone'][: len("timm_")] == "timm_"
                pretrained_model = mdetr_efficientnetB3(pretrained=pretrained, return_postprocessor=False)
            elif 'efficientnetB5' in load_path:
                assert mdetr_config['backbone'][: len("timm_")] == "timm_"
                pretrained_model = mdetr_efficientnetB5(pretrained=pretrained, return_postprocessor=False)
            elif 'clevr' in load_path:
                assert mdetr_config['backbone'] == 'resnet18'
                pretrained_model = mdetr_clevr(pretrained=pretrained, return_postprocessor=False)
            else:
                raise NotImplementedError

            # pretrained_params = {k: v for k, v in pretrained_model.named_parameters()}

            # Delete the extra encoder decoder layers in transformer state dict
            pretrained_state_dict = pretrained_model.state_dict()
            pretrained_state_dict_copy = pretrained_state_dict.copy()
            if mdetr_config.enc_layers < 6 or mdetr_config.dec_layers < 6:
                for k in pretrained_state_dict_copy.keys():
                    for layer_num in range(mdetr_config.enc_layers, 6):
                        if k.startswith(f'transformer.encoder.layers.{layer_num}'):
                            del pretrained_state_dict[k]
                    for layer_num in range(mdetr_config.dec_layers, 6):
                        if k.startswith(f'transformer.decoder.layers.{layer_num}'):
                            del pretrained_state_dict[k]

            # Delete keys that are not present in our MDETR model.
            for k in ['contrastive_align_projection_image', 'contrastive_align_projection_text']:
                # We do not have any contrastive losses.
                del pretrained_state_dict[k + '.weight']
                del pretrained_state_dict[k + '.bias']

            # Add keys used in our MDETR model but not present in the pretrained model.
            untrained_model_state_dict = self.model.state_dict()
            untrained_mdetr_keys = []

            if not self.use_pretrained_transformer:
                print('====> NOTE: Not using MDETR pretrained transformer. Only using pretrained image/text encoder.')
                for k in pretrained_state_dict.keys():
                    if k.startswith('transformer') and not k.startswith('transformer.text_encoder'):
                        pretrained_state_dict[k] = copy.deepcopy(untrained_model_state_dict[k])
                        untrained_mdetr_keys.append(k)

            for k in untrained_model_state_dict.keys():
                if 'adapter' in k:
                    untrained_mdetr_keys.append(k)
                    pretrained_state_dict[k] = copy.deepcopy(untrained_model_state_dict[k])

                elif k == 'transformer.CTRL_readout.weight':
                    untrained_mdetr_keys.append(k)
                    pretrained_state_dict[k] = copy.deepcopy(untrained_model_state_dict[k])

                elif mdetr_config.use_proprio and 'proprio_pos_emb' in k:
                    untrained_mdetr_keys.append(k)
                    pretrained_state_dict[k] = copy.deepcopy(untrained_model_state_dict[k])

                elif k == 'query_embed.weight':
                    # the sizes for query embedding may not match with the existing model
                    num_queries = untrained_model_state_dict[k].data.size(0)
                    assert pretrained_state_dict[k].size(0) == 100
                    pretrained_state_dict[k] = copy.deepcopy(pretrained_state_dict[k][:num_queries, :])
                

            # Load the pre-trained models (state dict is required since we have BN params / running mean, var)
            self.model.load_state_dict(pretrained_state_dict)

        augment_img = kwargs['augment_img']
        initial_resize = kwargs['initial_resize']
        use_crop_aug_during_eval = kwargs['use_crop_aug_during_eval']
        resize_tfs = [
            T.Resize(initial_resize),
            T.CenterCrop(224),
        ]
        normalize_tfs = [
            T.ToTensor(), # ToTensor() divides by 255
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

        # Get hand camera transforms
        resize_tfs = [
            T.Resize(initial_resize),
            T.CenterCrop(224),
        ]
        normalize_tfs = [
            T.ToTensor(), # ToTensor() divides by 255
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        shift_tfs = [ T.RandomCrop(224, padding=augment_padding, padding_mode='edge') ]
        self.hand_cam_transforms = T.Compose(resize_tfs + shift_tfs + normalize_tfs)
        self.hand_cam_eval_transforms = T.Compose(resize_tfs + normalize_tfs)
    
    def create_model_for_hand_camera_image(self, film_config):
        """Create model for hand camera images."""

        from mrest.utils.vision_models.mdetr_multiview_policy import create_model_for_hand_camera_image
        hand_camera_models_dict = create_model_for_hand_camera_image(self.mdetr_config, film_config)
        self.hand_cam_resnet_model = hand_camera_models_dict['hand_cam_resnet_model']
        self.hand_cam_spatial_conv = hand_camera_models_dict['hand_cam_spatial_conv']
        self.film_models = hand_camera_models_dict['film_models']


    def forward_static_image(self, inp,
                             tasks: Optional[List[str]] = None,
                             task_embs: Optional[torch.Tensor] = None,
                             task_descriptions: Optional[List[str]] = None,
                             obs_info: Optional[Dict[str, Any]] = None,) -> torch.Tensor:
        mdetr_output, text_embd = self.model.forward_images(inp, task_descriptions, obs_info['proprio_z'])

        return mdetr_output

    @property
    def use_multiview_batch(self):
        '''If True use run_on_multiview_batch from encoder_with_proprio.

        This function passes on all the images to the image encoder instead of separately doing it 
        in EncoderWithProprio.
        '''
        return True

    @property
    def use_film_for_hand_camera(self):
        attn_type = self.mdetr_config.multiview.attn_type
        return self.mdetr_config.multiview[attn_type]['hand_cam_resnet_film_config']['use']

    def forward_all_cameras(self,
                            camera_names: List[str],
                            tasks: Optional[List[str]] = None,
                            task_embs: Optional[torch.Tensor] = None,
                            task_descriptions: Optional[List[str]] = None,
                            obs_info: Optional[Dict[str, Any]] = None,
                            use_eval_transforms: bool = False,
                            device: str = 'cuda',
                            save_augs_and_return_cfg: Optional[Mapping[str, Any]] = None,
                            return_info_dict: bool = False,) -> torch.Tensor:
        imgs_by_camera = dict()
        other_camera_name, hand_camera_name = None, None
        for camera_name in camera_names:
            assert camera_name in ('left_cap2', 'eye_in_hand_90', 'robot0_eye_in_hand_90', 'front', 'front_rgb', 'wrist', 'wrist_rgb')
            if isinstance(obs_info[camera_name], torch.Tensor):
                imgs = obs_info[camera_name].numpy().astype(np.uint8)
            else:
                imgs = obs_info[camera_name].astype(np.uint8)
            hand_camera = 'hand' in camera_name or 'wrist' in camera_name
            if hand_camera:
                assert hand_camera_name is None, 'Cannot have more than 1 hand cameras'
                hand_camera_name = camera_name
                transforms = self.hand_cam_eval_transforms if use_eval_transforms else self.hand_cam_transforms
            else:
                assert other_camera_name is None, 'Cannot have more than 1 static camera'
                other_camera_name = camera_name
                transforms = self.eval_transforms if use_eval_transforms else self.transforms
            transf_imgs = [transforms(Image.fromarray(imgs[img_idx])).unsqueeze(0)
                           for img_idx in range(len(imgs))]
            transf_imgs = torch.cat(transf_imgs).to(device)
            imgs_by_camera[camera_name] = transf_imgs
        
        if self.attn_type == 'only_static':
            hs, text_embed = self.model.forward_images(
                imgs_by_camera[other_camera_name], task_descriptions, obs_info['proprio_z'])
            return hs

        elif self.attn_type == 'only_hand':
            hs, text_embed = self.model.forward_images(
                imgs_by_camera[hand_camera_name], task_descriptions, obs_info['proprio_z'])
            return hs

        elif self.attn_type == 'concat':

            hs, text_embed = self.model.forward_images(
                imgs_by_camera[other_camera_name], task_descriptions, obs_info['proprio_z'])

            if self.use_film_for_hand_camera:
                text_embed = text_embed.detach()
                film_outputs = []
                for layer_idx, num_blocks in enumerate(self.hand_cam_resnet_model.layers):
                    if self.film_models[layer_idx] is not None:
                        film_features = self.film_models[layer_idx](text_embed)
                    else:
                        film_features = None
                    film_outputs.append(film_features)
            else:
                film_outputs = None

            hand_imgs_z = self.hand_cam_resnet_model.forward(
                imgs_by_camera[hand_camera_name], film_features=film_outputs, flatten=False)
            hand_imgs_z = self.hand_cam_spatial_conv(hand_imgs_z)
            hand_imgs_z = reduce(hand_imgs_z, 'b c h w -> b c', 'mean')

            output = torch.cat([hs, hand_imgs_z], dim=1)

            # Return the CLS embedding
            return output

        else:
            raise NotImplementedError

    @property
    def output_embedding_size(self) -> int:
        return self.output_size

    def get_trainable_parameters(self):
        return self.parameters()

    def get_image_encoder_parameters(self, return_param_groups: bool = False):
        """Return image encoder parameters only. If the image encoder is frozen
        these parameters will not be updated."""
        if return_param_groups:
            def _text_filter(param_name):
                return 'text_encoder' in param_name
            def _backbone_adapter_filter(param_name):
                return 'backbone' in param_name and 'adapter' in param_name
            def _backbone_non_adapter_filter(param_name):
                return 'backbone' in param_name and 'adapter' not in param_name
            def _transformer_adapter_filter(param_name):
                return 'transformer' in param_name and 'adapter' in param_name
            def _transformer_non_adapter_filter(param_name):
                return 'transformer' in param_name and 'adapter' not in param_name

            text_encoder_params = []
            adapter_vision_params = []
            vision_backbone_params = []
            transformer_adapter_params = []
            transformer_non_adapter_params = []
            non_grouped_params = []
            print('MDETR policy found non-gouped params begin =>')
            for n, p in self.model.named_parameters():
                used = False
                if not p.requires_grad:
                    continue
                if _text_filter(n):
                    text_encoder_params.append(p); used = True
                if _backbone_adapter_filter(n):
                    assert not used
                    adapter_vision_params.append(p); used = True
                if _backbone_non_adapter_filter(n):
                    assert not used
                    vision_backbone_params.append(p); used = True
                if _transformer_adapter_filter(n):
                    assert not used
                    transformer_adapter_params.append(p); used = True
                if _transformer_non_adapter_filter(n):
                    assert not used
                    transformer_non_adapter_params.append(p); used = True

                if not used:
                    print(f'\t\t: {n}')
                    non_grouped_params.append(p)

            print('MDETR policy found non-gouped params end <=')
            params_dict = {
                'text_encoder_params': text_encoder_params,
                'adapter_vision_params': adapter_vision_params,
                'vision_backbone_params': vision_backbone_params,
                'transformer_adapter_params': transformer_adapter_params,
                'transformer_non_adapter_params': transformer_non_adapter_params,
                'non_grouped_params': non_grouped_params,
            }
            if self.attn_type == 'only_static' or self.attn_type == 'only_hand':
                pass

            elif self.attn_type == 'concat':
                hand_camera_params = [p for p in itertools.chain(self.hand_cam_resnet_model.parameters(),
                                                                 self.hand_cam_spatial_conv.parameters(),
                                                                 )]
                if self.use_film_for_hand_camera:
                    hand_camera_params.extend([p for p in self.film_models.parameters()])

                params_dict['hand_camera_visual_params'] = hand_camera_params

            return params_dict

        else:
            return self.parameters()

    def get_nonimage_encoder_parameters(self):
        """Return any non-image encoder parameters.

        Even if the image encoder is frozen these parameters will still be updated.
        This is used for down_projection layers or task embeddings (e.g. FiLM modules).
        """
        # raise NotImplementedError
        return []

    def load_non_adapter_pretrained_state_dict(self, pretrained_state_dict):
        # Delete keys that are not present in our MDETR model.
        # for k in ['contrastive_align_projection_image', 'contrastive_align_projection_text']:
        #     # We do not have any contrastive losses.
        #     del pretrained_state_dict[k + '.weight']
        #     del pretrained_state_dict[k + '.bias']

        # Add keys used in our MDETR model but not present in the pretrained model.
        curr_model_state_dict = self.state_dict()
        curr_mdetr_keys = []

        if not self.use_pretrained_transformer:
            print('====> NOTE: Not using MDETR pretrained transformer. Only using pretrained image/text encoder.')
            for k in pretrained_state_dict.keys():
                if k.startswith('transformer') and not k.startswith('transformer.text_encoder'):
                    pretrained_state_dict[k] = copy.deepcopy(curr_model_state_dict[k])
                    curr_mdetr_keys.append(k)

        for k in curr_model_state_dict.keys():
            if 'adapter' in k:
                curr_mdetr_keys.append(k)
                pretrained_state_dict[k] = copy.deepcopy(curr_model_state_dict[k])

        # Load the pre-trained models (state dict is required since we have BN params / running mean, var)
        self.load_state_dict(pretrained_state_dict)

