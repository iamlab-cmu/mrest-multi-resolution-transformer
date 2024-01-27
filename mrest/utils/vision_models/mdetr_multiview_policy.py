import copy
import itertools
import numpy as np
import pprint

from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from einops import rearrange

from PIL import Image
import torchvision.models as models
import torchvision.transforms as T

from mrest.utils.mdetr.mdetr_model import MDETR
from mrest.utils.mdetr.backbone import build_backbone
from mrest.utils.mdetr.transformer import build_transformer, TransformerEncoderLayer, TransformerEncoder
from mrest.utils.mdetr.model_hub import mdetr_resnet101, mdetr_efficientnetB3, mdetr_efficientnetB5, mdetr_clevr
from mrest.utils.mdetr.util.misc import NestedTensor
from mrest.utils.vision_models.resnet_base import resnet18

from mrest.utils.mdetr.position_encoding import build_position_encoding


def create_model_for_hand_camera_image(mdetr_config, film_config):
    """Create model for hand camera images."""

    weights = 'IMAGENET1K_V1'
    hand_cam_resnet_model = resnet18(
        weights=weights,
        film_config=film_config if film_config.use else None)
    hand_cam_spatial_conv = nn.Conv2d(512, mdetr_config.hidden_dim, kernel_size=1, stride=1)

    # FiLM config
    if film_config.use:
        film_models = []
        for layer_idx, num_blocks in enumerate(hand_cam_resnet_model.layers):
            if layer_idx in film_config.use_in_layers:
                num_planes = hand_cam_resnet_model.film_planes[layer_idx]
                film_model_layer = nn.Linear(
                    film_config.task_embedding_dim, num_blocks * 2 * num_planes)
            else:
                film_model_layer = None
            film_models.append(film_model_layer)
        film_models = nn.ModuleList(film_models)

    # NOTE: This works for ResNet backbones but should check if same
    # template applies to other backbone architectures
    hand_cam_resnet_model.fc = nn.Identity()
    hand_cam_resnet_model.avgpool = nn.Identity()

    return {
        'hand_cam_resnet_model': hand_cam_resnet_model, 
        'hand_cam_spatial_conv': hand_cam_spatial_conv,
        'film_models': film_models,
    }


class ImageAugmentations:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        resize_tfs = [
                T.Resize(cfg['initial_resize']),
                T.CenterCrop(224),
            ]
        transforms = resize_tfs

        # Normalize at the end
        normalize_tfs = [
            T.ToTensor(), # ToTensor() divides by 255
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

        if cfg['random_crop']:
            shift_tfs = [ T.RandomCrop(224, padding=cfg['augment_padding'], padding_mode='edge') ]
            transforms = transforms + shift_tfs
        
        self.jitter_prob = None
        if cfg['color_jitter']:
            # color_jitter_tfs = [T.ColorJitter(brightness=(0.4, 0.8), contrast=(0.4, 0.6), saturation=(0.4, 0.6), hue=(0.3, 0.5))]
            # color_jitter_tfs = [T.ColorJitter(brightness=(0.8, 0.8), contrast=(0.8, 0.8), saturation=(0.4, 0.6), hue=(0.3, 0.5))]
            color_jitter_tfs = [T.ColorJitter(brightness=(0.4, 0.8), contrast=(0.4, 0.8), saturation=(0.4, 0.6), hue=(0.0, 0.5))]
            if cfg['stochastic_jitter']:
                # No jitter in these transforms
                self.transforms_no_jitter = T.Compose(transforms + normalize_tfs)
                transforms = transforms + color_jitter_tfs
                self.jitter_prob = 0.5
            else:
                transforms = transforms + color_jitter_tfs
                self.jitter_prob = 1.0
        
        transforms = transforms + normalize_tfs
        self.transforms = T.Compose(transforms)
    
    def __call__(self, x) -> Any:
        return self.apply(x)
    
    def apply(self, x):
        if self.jitter_prob is not None:
            if np.random.uniform(0, 1) < self.jitter_prob:
                return self.transforms(x)
            else:
                return self.transforms_no_jitter(x)

        return self.transforms(x)
    
    def undo_normalizations(self, x):
        mean, std = self.transforms.transforms[-1].mean, self.transforms.transforms[-1].std
        mean = torch.FloatTensor(mean).to(x.device).view(-1, 1, 1)
        std = torch.FloatTensor(std).to(x.device).view(-1, 1, 1)
        out = ((x * std + mean) * 255).long().cpu().numpy()
        return out


class MDETRMultiViewPolicy(nn.Module):

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


        if self.attn_type == 'attn_self':

            self.multiview_output_emb = nn.Embedding(1, mdetr_config.hidden_dim)

            # Learnable embeddings for cameras (not using them for now)
            self.static_cam_emb = nn.Embedding(1, mdetr_config.hidden_dim)
            self.hand_cam_emb = nn.Embedding(1, mdetr_config.hidden_dim)

            self.output_size = mdetr_config.multiview.attn_self.hidden_dim
            multiview_encoder_layer = TransformerEncoderLayer(
                mdetr_config.multiview.attn_self.hidden_dim,
                mdetr_config.multiview.attn_self.nheads,
                mdetr_config.multiview.attn_self.dim_feedforward,
                mdetr_config.multiview.attn_self.dropout,
                mdetr_config.multiview.attn_self.nonlinearity,
                mdetr_config.multiview.attn_self.pre_norm,)
            multiview_encoder_norm = (nn.LayerNorm(args.mdetr_config.multiview.attn_self.hidden_dim)
                if mdetr_config.pre_norm else None)
            self.multiview_encoder = TransformerEncoder(
                multiview_encoder_layer,
                mdetr_config.enc_layers,
                multiview_encoder_norm)

            # Create Hand Camera model
            film_config = mdetr_config.multiview.attn_self.hand_cam_resnet_film_config
            self.create_model_for_hand_camera_image(film_config)

        elif self.attn_type == 'attn_cross':

            attn_cross_config = mdetr_config.multiview.attn_cross

            # Learnable embeddings for cameras
            self.static_cam_emb = nn.Embedding(1, mdetr_config.hidden_dim)
            self.hand_cam_emb = nn.Embedding(1, mdetr_config.hidden_dim)

            if 'hand_cam_position_encoding' in attn_cross_config and attn_cross_config.hand_cam_position_encoding.use:
                self.hand_cam_img_position_emb = build_position_encoding(attn_cross_config.hand_cam_position_encoding)
            else:
                self.hand_cam_img_position_emb = None

            self.output_size = mdetr_config.multiview.attn_self.hidden_dim
            for layer_idx, layer in enumerate(self.model.transformer.encoder.layers):
                if layer_idx in attn_cross_config.cam1to2_config.layers:
                    layer.create_multiview_models(attn_cross_config.cam1to2_config)

            # Create Hand Camera model
            film_config = mdetr_config.multiview.attn_cross.hand_cam_resnet_film_config
            self.create_model_for_hand_camera_image(film_config)

        elif self.attn_type == 'attn_cross_2':

            attn_cross_config = mdetr_config.multiview.attn_cross_2
            cam2_transformer_cfg = mdetr_config.multiview.attn_cross_2.cam2_transformer

            # Learnable embeddings for cameras
            self.static_cam_emb = nn.Embedding(1, mdetr_config.hidden_dim)
            self.hand_cam_emb = nn.Embedding(1, mdetr_config.hidden_dim)

            if 'hand_cam_position_encoding' in attn_cross_config and attn_cross_config.hand_cam_position_encoding.use:
                self.hand_cam_img_position_emb = build_position_encoding(attn_cross_config.hand_cam_position_encoding)
            else:
                self.hand_cam_img_position_emb = None

            # Embedding from  both cameras
            self.output_size = mdetr_config.multiview.attn_cross_2.hidden_dim * 2
            for layer_idx, layer in enumerate(self.model.transformer.encoder.layers):
                if layer_idx in attn_cross_config.cam1to2_config.layers:
                    layer.create_multiview_models(attn_cross_config.cam1to2_config)

            # Create Hand Camera model
            film_config = mdetr_config.multiview.attn_cross_2.hand_cam_resnet_film_config
            self.create_model_for_hand_camera_image(film_config)

            hand_cam_encoder_layer = TransformerEncoderLayer(
                cam2_transformer_cfg.hidden_dim,
                cam2_transformer_cfg.nheads,
                cam2_transformer_cfg.dim_feedforward, 
                cam2_transformer_cfg.dropout, 
                cam2_transformer_cfg.activation,
                cam2_transformer_cfg.pre_norm,)
            encoder_norm = nn.LayerNorm(cam2_transformer_cfg.hidden_dim) if cam2_transformer_cfg.pre_norm else None
            self.hand_cam_CTRL = nn.Embedding(1, cam2_transformer_cfg.hidden_dim)
            self.hand_cam_transformer_encoder = TransformerEncoder(
                hand_cam_encoder_layer, 
                cam2_transformer_cfg.enc_layers,
                encoder_norm,)
            for layer_idx, layer in enumerate(self.hand_cam_transformer_encoder.layers):
                layer.create_multiview_models(attn_cross_config.cam2to1_config)


        elif self.attn_type == 'only_static':
            # Use only static camera
            assert mdetr_config.use_control_readout_emb, 'Should set control embedding to True'
            assert mdetr_config.enc_layers == 6, 'Use same encoders as before (for reproducibility)'
            self.output_size = 256

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
                
                elif 'cam1to2' in k:
                    untrained_mdetr_keys.append(k)
                    pretrained_state_dict[k] = copy.deepcopy(untrained_model_state_dict[k])


            # Load the pre-trained models (state dict is required since we have BN params / running mean, var)
            self.model.load_state_dict(pretrained_state_dict)
        image_augmentations_cfg = kwargs['image_augmentations']
        self.static_cam_augs = ImageAugmentations(image_augmentations_cfg['left_cap2']['train'])
        self.static_cam_eval_augs = ImageAugmentations(image_augmentations_cfg['left_cap2']['train'])
        self.hand_cam_augs = ImageAugmentations(image_augmentations_cfg['eye_in_hand_90']['train'])
        self.hand_cam_eval_augs = ImageAugmentations(image_augmentations_cfg['eye_in_hand_90']['eval'])

    def create_model_for_hand_camera_image(self, film_config):
        """Create model for hand camera images."""

        weights = 'IMAGENET1K_V1'
        self.hand_cam_resnet_model = resnet18(
            weights=weights,
            film_config=film_config if film_config.use else None)
        self.hand_cam_spatial_conv = nn.Conv2d(512, self.mdetr_config.hidden_dim, kernel_size=1, stride=1)

        # FiLM config
        if film_config.use:
            film_models = []
            for layer_idx, num_blocks in enumerate(self.hand_cam_resnet_model.layers):
                if layer_idx in film_config.use_in_layers:
                    num_planes = self.hand_cam_resnet_model.film_planes[layer_idx]
                    film_model_layer = nn.Linear(
                        film_config.task_embedding_dim, num_blocks * 2 * num_planes)
                else:
                    film_model_layer = None
                film_models.append(film_model_layer)
            self.film_models = nn.ModuleList(film_models)

        # NOTE: This works for ResNet backbones but should check if same
        # template applies to other backbone architectures
        self.hand_cam_resnet_model.fc = nn.Identity()
        self.hand_cam_resnet_model.avgpool = nn.Identity()


    def forward_static_image(self, inp,
                             tasks: Optional[List[str]] = None,
                             task_embs: Optional[torch.Tensor] = None,
                             task_descriptions: Optional[List[str]] = None,
                             obs_info: Optional[Dict[str, Any]] = None,) -> torch.Tensor:
        mdetr_output, text_embd = self.model.forward_images(inp, task_descriptions, obs_info['proprio_z'])

        return mdetr_output

    def encode_multiple_views(self, obs_dict, camera_names):
        enc_imgs = []
        for camera_name in camera_names:
            enc_imgs.append(obs_dict[camera_name])
        inp = torch.cat(enc_imgs, axis=1)
        # return self.multiview_encoder(inp)
        return inp
    
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
        if attn_type not in ('attn_self', 'attn_cross', 'attn_cross_2'):
            return False
        return self.mdetr_config.multiview[attn_type]['hand_cam_resnet_film_config']['use']

    def forward_all_cameras(self,
                            camera_names: List[str],
                            tasks: Optional[List[str]] = None,
                            task_embs: Optional[torch.Tensor] = None,
                            task_descriptions: Optional[List[str]] = None,
                            obs_info: Optional[Dict[str, Any]] = None,
                            use_eval_transforms: bool = False,
                            device: str = 'cuda',
                            save_augs_and_return_cfg: Optional[Mapping[str, Any]] = None,) -> torch.Tensor:
        imgs_by_camera = dict()
        other_camera_name, hand_camera_name = None, None
        for camera_name in camera_names:
            assert camera_name in ('left_cap2', 'eye_in_hand_90', 'eye_in_hand_45', 'robot0_eye_in_hand_90',
                                   'front', 'front_rgb', 'wrist', 'wrist_rgb', 'static', 'hand')
            if isinstance(obs_info[camera_name], torch.Tensor):
                imgs = obs_info[camera_name].numpy().astype(np.uint8)
            else:
                imgs = obs_info[camera_name].astype(np.uint8)
            hand_camera = 'hand' in camera_name or 'wrist' in camera_name
            if hand_camera:
                assert hand_camera_name is None, 'Cannot have more than 1 hand cameras'
                transforms = self.hand_cam_eval_augs if use_eval_transforms else self.hand_cam_augs
                hand_camera_name = camera_name
            else:
                assert other_camera_name is None, 'Cannot have more than 1 static camera'
                other_camera_name = camera_name
                transforms = self.static_cam_eval_augs if use_eval_transforms else self.static_cam_augs

            transf_imgs = [transforms(Image.fromarray(imgs[img_idx])).unsqueeze(0)
                           for img_idx in range(len(imgs))]
            transf_imgs = torch.cat(transf_imgs).to(device)
            imgs_by_camera[camera_name] = transf_imgs
        
        # TODO(Mohit): Save images after augmentations (just to verify)
        if save_augs_and_return_cfg is not None and save_augs_and_return_cfg['use']:
            # path_to_save = save_augs_and_return_cfg['path']
            # epoch = save_augs_and_return_cfg['epoch']
            num_images = save_augs_and_return_cfg['num_images']
            img_after_aug_static = self.static_cam_augs.undo_normalizations(imgs_by_camera[other_camera_name][:num_images])
            img_after_aug_hand = self.hand_cam_augs.undo_normalizations(imgs_by_camera[hand_camera_name][:num_images])
            tasks = tasks[:num_images]
            return {
                'hand_imgs_after_aug': img_after_aug_hand,
                'static_imgs_after_aug': img_after_aug_static,
                'tasks': tasks
            }


        if self.attn_type == 'only_static':
            hs, text_embed = self.model.forward_images(
                imgs_by_camera[other_camera_name], task_descriptions, obs_info['proprio_z'])
            return hs, {}

        elif self.attn_type == 'attn_self':

            static_img_mdetr_output = self.model.forward_images(
                imgs_by_camera[other_camera_name], task_descriptions, obs_info['proprio_z'])
            static_img_lang_z = static_img_mdetr_output['img_memory']
            static_img_lang_mask = static_img_mdetr_output['mask']
            static_img_lang_pos_emb = static_img_mdetr_output['pos_embed']

            if self.use_film_for_hand_camera:
                task_embs = static_img_mdetr_output['text_embedding'].detach()
                film_outputs = []
                for layer_idx, num_blocks in enumerate(self.hand_cam_resnet_model.layers):
                    if self.film_models[layer_idx] is not None:
                        film_features = self.film_models[layer_idx](task_embs)
                    else:
                        film_features = None
                    film_outputs.append(film_features)
            else:
                film_outputs = None

            hand_imgs_z = self.hand_cam_resnet_model.forward(
                imgs_by_camera[hand_camera_name], film_features=film_outputs, flatten=False)
            hand_imgs_z = self.hand_cam_spatial_conv(hand_imgs_z)

            B, D = hand_imgs_z.shape[:2]
            device = static_img_lang_z.device

            hand_imgs_z = rearrange(hand_imgs_z, 'b d h w -> (h w) b d')
            hand_cam_pos_emb = self.hand_cam_emb.weight.view(1, 1, -1).repeat(
                hand_imgs_z.size(0), B, 1)
            static_cam_pos_emb = self.static_cam_emb.weight.view(1, 1, -1).repeat(
                static_img_mdetr_output['text_memory_start_idx'], B, 1
            )
            static_cam_pos_emb = torch.cat((
                static_cam_pos_emb,
                torch.zeros((static_img_mdetr_output['text_memory'].size(0), B, D)).to(device),
            ))

            static_img_lang_pos_emb += static_cam_pos_emb

            # Get position embeddings
            pos_embed = torch.cat((torch.zeros(1, B, D, device=device), hand_cam_pos_emb, static_img_lang_pos_emb))

            CLS = self.multiview_output_emb.weight.view(1, 1, -1).repeat(1, B, 1)
            multiview_img_z = torch.cat([CLS, hand_imgs_z, static_img_lang_z], dim=0)

            mask = torch.cat((torch.zeros(B, hand_imgs_z.size(0) + 1).bool().to(device),  static_img_lang_mask), dim=1)
            output = self.multiview_encoder(multiview_img_z, src_key_padding_mask=mask, pos=pos_embed)

            # Return the CLS embedding
            return output[0], {}

        elif self.attn_type == 'attn_cross':
            # Get unimodal I_3, and text embedding
            unimodal_memory_cache = self.model.forward_unimodal_input(
                imgs_by_camera[other_camera_name], task_descriptions, obs_info['proprio_z'],
            )
            # Get unimodal I_h embedding
            if self.use_film_for_hand_camera:
                task_embs = unimodal_memory_cache['text_embedding'].detach()
                film_outputs = []
                for layer_idx, num_blocks in enumerate(self.hand_cam_resnet_model.layers):
                    if self.film_models[layer_idx] is not None:
                        film_features = self.film_models[layer_idx](task_embs)
                    else:
                        film_features = None
                    film_outputs.append(film_features)
            else:
                film_outputs = None
            hand_imgs_z = self.hand_cam_resnet_model.forward(
                imgs_by_camera[hand_camera_name], film_features=film_outputs, flatten=False)
            hand_imgs_z = self.hand_cam_spatial_conv(hand_imgs_z)
            B, D = hand_imgs_z.shape[:2]
            device = hand_imgs_z.device
            hand_imgs_z = rearrange(hand_imgs_z, 'b d h w -> (h w) b d')
            # hand_cam_pos_emb = self.hand_cam_emb.weight.view(1, 1, -1).repeat(
            #     hand_imgs_z.size(0), B, 1)
        
            output = unimodal_memory_cache['src']
            mask = unimodal_memory_cache['mask']
            pos_embed = unimodal_memory_cache['pos_embed']

            cross_attn_config = self.mdetr_config.multiview.attn_cross
            for layer_idx, layer in enumerate(self.model.transformer.encoder.layers):
                if layer_idx in cross_attn_config.cam1to2_config.layers:
                    output = layer(output, src_key_padding_mask=mask, pos=pos_embed, src_other_cam=hand_imgs_z)
                else:
                    output = layer(output, src_key_padding_mask=mask, pos=pos_embed)
            if self.model.transformer.encoder.norm is not None:
                output = self.model.transformer.encoder.norm(output)
            return output[0], {}

        elif self.attn_type == 'attn_cross_2':
            # Get unimodal I_3, and text embedding
            unimodal_memory_cache = self.model.forward_unimodal_input(
                imgs_by_camera[other_camera_name], task_descriptions, obs_info['proprio_z'],
            )
            # Get unimodal I_h embedding
            if self.use_film_for_hand_camera:
                task_embs = unimodal_memory_cache['text_embedding'].detach()
                film_outputs = []
                for layer_idx, num_blocks in enumerate(self.hand_cam_resnet_model.layers):
                    if self.film_models[layer_idx] is not None:
                        film_features = self.film_models[layer_idx](task_embs)
                    else:
                        film_features = None
                    film_outputs.append(film_features)
            else:
                film_outputs = None
            hand_imgs_z = self.hand_cam_resnet_model.forward(
                imgs_by_camera[hand_camera_name], film_features=film_outputs, flatten=False)
            hand_imgs_z = self.hand_cam_spatial_conv(hand_imgs_z)
            B, D = hand_imgs_z.shape[:2]
            device = hand_imgs_z.device
            cross_attn_config = self.mdetr_config.multiview[self.attn_type]

            # Optionally add position encoding to ResNet outputs
            if self.hand_cam_img_position_emb is not None:
                hand_mask = torch.zeros((B, hand_imgs_z.size(2), hand_imgs_z.size(3))).bool().to(device)
                hand_imgs_z_pos = self.hand_cam_img_position_emb.forward_with_tensor_mask(hand_imgs_z, hand_mask)

            hand_imgs_z = rearrange(hand_imgs_z, 'b d h w -> (h w) b d')
            hand_cam_CTRL = self.hand_cam_CTRL.weight.view(1, 1, -1).repeat(1, B, 1)
            hand_imgs_z = torch.cat([hand_cam_CTRL, hand_imgs_z], dim=0)
            if self.hand_cam_img_position_emb is not None:
                hand_imgs_z_pos = rearrange(hand_imgs_z_pos, 'b d h w -> (h w) b d')
                hand_imgs_z_pos = torch.cat([torch.zeros_like(hand_cam_CTRL).to(device), hand_imgs_z_pos])
                # Add camera type embedding to position embedding. 
                # Check if this useful? Hypothesis: since we use sine embeddings for static and in-hand camera
                # we do need someway to differentiate across these position embeddings?
                if cross_attn_config.hand_cam_position_encoding.add_hand_cam_embedding:
                    hand_cam_pos_emb = self.hand_cam_emb.weight.view(1, 1, -1).repeat(
                        hand_imgs_z.size(0), B, 1)
                    hand_imgs_z_pos = hand_imgs_z_pos + hand_cam_pos_emb
                # Add position embedding to hand_img_z. We do this here so that we can use these pos embeddings
                # before cross-attention
                hand_imgs_z = hand_imgs_z + hand_imgs_z_pos
        
            output = unimodal_memory_cache['src']
            mask = unimodal_memory_cache['mask']
            pos_embed = unimodal_memory_cache['pos_embed']

            attn_weights = {}
            for layer_idx, layer in enumerate(self.model.transformer.encoder.layers):
                if layer_idx in cross_attn_config.cam1to2_config.layers:
                    output = layer(output, src_key_padding_mask=mask, pos=pos_embed, src_other_cam=hand_imgs_z)

                    hand_cam_layer = self.hand_cam_transformer_encoder.layers[layer_idx - 3]
                    hand_imgs_z = hand_cam_layer(hand_imgs_z, src_other_cam=output)
                else:
                    output, attn_weights_dict = layer(output, src_key_padding_mask=mask, pos=pos_embed, return_attn_weights=True)
                    # output = layer(output, src_key_padding_mask=mask, pos=pos_embed)
                    attn_weights[layer_idx] = attn_weights_dict

            info_dict = {
                'transformer_attn_weights': attn_weights,
            }
            return torch.cat([output[0], hand_imgs_z[0]], dim=-1), info_dict

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
            if self.attn_type == 'attn_self':
                hand_camera_params = [p for p in itertools.chain(self.hand_cam_resnet_model.parameters(),
                                                                 self.hand_cam_spatial_conv.parameters(),
                                                                 self.multiview_output_emb.parameters(),
                                                                 self.hand_cam_emb.parameters(),
                                                                 self.static_cam_emb.parameters())]
                if self.use_film_for_hand_camera:
                    hand_camera_params.extend([p for p in self.film_models.parameters()])

                multiview_params = [p for p in self.multiview_encoder.parameters()]
                params_dict['multiview_encoder_params'] = multiview_params
                params_dict['hand_camera_params'] = hand_camera_params

            elif self.attn_type == 'attn_cross':
                hand_camera_params = [p for p in itertools.chain(self.hand_cam_resnet_model.parameters(),
                                                                 self.hand_cam_spatial_conv.parameters(),
                                                                 self.hand_cam_emb.parameters(),
                                                                 self.static_cam_emb.parameters(),
                                                                 )]
                if self.use_film_for_hand_camera:
                    hand_camera_params.extend([p for p in self.film_models.parameters()])

                params_dict['hand_camera_params'] = hand_camera_params
            elif self.attn_type == 'attn_cross_2':
                hand_camera_params = [p for p in itertools.chain(self.hand_cam_resnet_model.parameters(),
                                                                 self.hand_cam_spatial_conv.parameters(),
                                                                 self.hand_cam_emb.parameters(),
                                                                 self.static_cam_emb.parameters(),
                                                                 )]
                if self.use_film_for_hand_camera:
                    hand_camera_params.extend([p for p in self.film_models.parameters()])

                params_dict['hand_camera_visual_params'] = hand_camera_params
                params_dict['hand_camera_transformer_params'] = [p for p in itertools.chain(
                    self.hand_cam_transformer_encoder.parameters(),
                    self.hand_cam_img_position_emb.parameters() if self.hand_cam_img_position_emb is not None else [],
                    self.hand_cam_CTRL.parameters(),)]
                    
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
