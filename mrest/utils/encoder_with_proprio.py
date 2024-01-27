from typing import Any, Dict, List, Mapping, Optional, Union
import copy
import numpy as np
import re

import torch
import torch.nn as nn

from PIL import Image
import torchvision.models as models
import torchvision.transforms as T
from omegaconf import DictConfig, ListConfig

from mrest.utils.torch_utils import get_nonlinearity_from_str
from mrest.utils.gaussian_mlp import MLP
from mrest.utils.language.task_language_embedding_model import TaskEmbeddingController


class ProprioEncoder(nn.Module):
    def __init__(self,
                 proprio_inp_size: int,
                 hidden_sizes: Optional[List] = None,
                 final_emb_size: Optional[int] = None,
                 nonlinearity: str = 'relu') -> None:
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.final_emb_size = final_emb_size
        self.nonlinearity = nonlinearity

        proprio_sizes = [proprio_inp_size]
        if self.hidden_sizes and len(self.hidden_sizes) > 0:
            proprio_sizes.extend(list(self.hidden_sizes))

        assert final_emb_size is not None
        if final_emb_size == 0:
            self.proprio_mlp = nn.Identity()
        else:
            proprio_sizes.append(final_emb_size)

        modules = []
        for i in range(len(proprio_sizes) - 1):
            modules.append(nn.Linear(proprio_sizes[i], proprio_sizes[i + 1]))
            if i < len(proprio_sizes) - 2:
                modules.append(get_nonlinearity_from_str(nonlinearity))
        self.proprio_mlp = nn.Sequential(*modules)
        self.proprio_embedding_size = proprio_sizes[-1]

    def forward(self, inp: torch.Tensor):
        if inp.size(1) != self.proprio_mlp[0].in_features:
            inp = torch.cat([inp, inp[:, :1]], dim=1)
        out = self.proprio_mlp(inp)
        return out

    @property
    def output_embedding_size(self):
        return self.proprio_embedding_size


class EncoderWithProprio:
    def __init__(self,
                 image_encoder,
                 proprio_encoder: ProprioEncoder,
                 policy: MLP,
                 language_emb_model: TaskEmbeddingController,
                 camera_names: List[str],
                 proprio_keys: List[str],
                 finetune_image_encoder: bool,
                 optimizer_cfg: Mapping[str, Any],
                 randomize_task_description_sampling: bool = False,
                 append_object_mask=None,
                 device: str = 'cuda') -> None:
        super().__init__()

        self.image_encoder = image_encoder
        self.proprio_encoder = proprio_encoder
        self.policy = policy
        # NOTE: language_emb_model can be None
        self.language_emb_model = language_emb_model
        self.camera_names = camera_names
        self.proprio_keys = proprio_keys
        assert isinstance(finetune_image_encoder, bool)
        self.freeze_image_encoder = not finetune_image_encoder
        self.device = device
        self.randomize_task_description_sampling = randomize_task_description_sampling
        self.append_object_mask = append_object_mask

        self._optimizer_cfg = optimizer_cfg

        # TODO(Mohit): Create a task embedding module here that takes task
        # names or ids and outputs task embeddings, can be random embeddings
        # or learned embeddings?

    def send_to_device(self, device: Optional[Union[str, torch.device]] = None):
        if device is None:
            device = self.device
        self.image_encoder = self.image_encoder.to(device)
        self.proprio_encoder = self.proprio_encoder.to(device)
        self.policy.send_to_device(device)

    def get_total_parameter_size(self, params):
        trainable_size = np.sum([np.prod(p.shape) for p in params if p.requires_grad])
        all_params_size = np.sum([np.prod(p.shape) for p in params])
        return {'all': all_params_size, 'trainable': trainable_size}

    def get_printable_parameter_size(self, param_size):
        if param_size > 1e9:
            return f'{float(param_size) / 1e9:.2f} G ({param_size})'
        elif param_size > 1e6:
            return f'{float(param_size) / 1e6:.2f} M ({param_size})'
        elif param_size > 1e3:
            return f'{float(param_size) / 1e3:.2f} K ({param_size})'
        else:
            return f'{float(param_size):.2f} '

    def print_parameter_groups(self, param_groups):
        from prettytable import PrettyTable
        params_table = PrettyTable()
        params_table.field_names = ["Type", "All params", "Train Params", "% Train", "% Train (non-zero LR)"]
        for param_type, params in param_groups.items():
            params_size_by_type = self.get_total_parameter_size(params)
            all_params = self.get_printable_parameter_size(params_size_by_type['all'])
            trainable_params = self.get_printable_parameter_size(params_size_by_type['trainable'])
            train_percent = (params_size_by_type['trainable'] * 100.) / (params_size_by_type['all'] + 1e-10)
            train_percent_mult_lr = train_percent
            if param_type not in self._optimizer_cfg['param_groups']:
                raise ValueError(f'No lr specified for param group: {param_type}')
            if self._optimizer_cfg['param_groups'][param_type]['lr'] <= 1e-10:
                train_percent_mult_lr = 0.0
            params_table.add_row([param_type, all_params, trainable_params, f'{train_percent:.4f}', f'{train_percent_mult_lr:.4f}'])

        print(params_table)

    def print_parameters(self, proprio_params, policy_params, lang_parameters,
                         non_image_vision_encoder_params, image_encoder_params=None):

        from prettytable import PrettyTable

        all_params = [proprio_params, policy_params, lang_parameters,
                      non_image_vision_encoder_params, image_encoder_params]
        param_types = ['proprio', 'policy', 'lang', 'image_enc_nonimage', 'image_enc_image']
        params_table = PrettyTable()
        params_table.field_names = ["Type", "All params", "Train Params", "% Train"]
        for param_type, params in zip(param_types, all_params):
            params_size_by_type = self.get_total_parameter_size(params)
            all_params = self.get_printable_parameter_size(params_size_by_type['all'])
            trainable_params = self.get_printable_parameter_size(params_size_by_type['trainable'])
            train_percent = (params_size_by_type['trainable'] * 100.) / (params_size_by_type['all'] + 1e-10)

            params_table.add_row([param_type, all_params, trainable_params, f'{train_percent:.4f}'])
        print(params_table)

    def get_trainable_parameters(self, pretty_print: bool = True, return_param_groups: bool = False):
        proprio_parameters = list(self.proprio_encoder.parameters())
        policy_parameters = list(self.policy.trainable_params())
        assert isinstance(proprio_parameters, list), "Proprio parameters should be a list not a generator"
        assert isinstance(policy_parameters, list), "Policy parameters should be a list not a generator"
        lang_parameters = []
        if self.language_emb_model:
            lang_parameters = list(self.language_emb_model.get_trainable_parameters())
            assert isinstance(lang_parameters, list), (
                "Language parameters should be a list not a generator")

        non_image_vision_encoder_params = list(self.image_encoder.get_nonimage_encoder_parameters())

        param_groups = {}
        if return_param_groups:
            param_groups['proprio_params'] = proprio_parameters
            param_groups['policy_params'] = policy_parameters
            param_groups['separate_lang_encoder_params'] = lang_parameters
            param_groups['non_image_vision_encoder_params'] = non_image_vision_encoder_params

        else:
            non_image_params = (proprio_parameters + policy_parameters
                + lang_parameters + non_image_vision_encoder_params)

        image_encoder_params = []
        if not self.freeze_image_encoder:
            if return_param_groups:
                image_encoder_params_dict = self.image_encoder.get_image_encoder_parameters(
                    return_param_groups=True)
                for k, v in image_encoder_params_dict.items():
                    assert k not in param_groups
                    param_groups[k] = v
            else:
                image_encoder_params = list(
                    self.image_encoder.get_image_encoder_parameters(
                        return_param_groups=return_param_groups))

        # Pretty print
        if pretty_print:
            if return_param_groups:
                self.print_parameter_groups(param_groups)
            else:
                self.print_parameters(proprio_parameters, policy_parameters, lang_parameters,
                                      non_image_vision_encoder_params, image_encoder_params=image_encoder_params)

        if return_param_groups:
            return param_groups
        else:
            return image_encoder_params + non_image_params

    def save_task_descriptions_for_lazy_envs(self, env_callable, env_suite_type,
                                             train_envs: List[str], heldout_envs: List[str]):
        self.task_descriptions_by_task_name = dict()
        env_types = ['train', 'test']
        for env_type, env_set in zip(env_types, [train_envs, heldout_envs]):
            env_descriptions_by_name = dict()
            for env_info in env_set:
                env = env_callable(env_info, env_type == 'train')

                if hasattr(env, 'task_descriptions'):
                    task_descriptions = env.task_descriptions
                else:
                    task_descriptions = env.env.task_descriptions

                if isinstance(env_info, str):
                    env_name = env_info
                else:
                    env_name = env_info.task_variation_name

                env_descriptions_by_name[env_name] = task_descriptions
                # assert self.task_descriptions_by_task_name.get(env_name) is None, (
                #     'Duplicate env in train and heldout envs')
                self.task_descriptions_by_task_name[env_name] = copy.deepcopy(task_descriptions)

                if env_suite_type == 'rlbench':
                    env.close()

            if self.language_emb_model:
                self.language_emb_model.cache_task_embedding_for_tasks(env_descriptions_by_name)
                print(f'Did create task embeddings for {env_type}: {len(env_descriptions_by_name)}')


    def save_task_descriptions(self, envs_dict):
        self.task_descriptions_by_task_name = dict()
        # env_types = ['train', 'test']
        # for env_type, env_set in zip(env_types, [train_envs, heldout_envs]):
        for env_type, env_set in envs_dict.items():
            env_descriptions_by_name = dict()
            for env in env_set:
                env_name = env.env_id
                task_descriptions = env.env.task_descriptions
                # NOTE: For button-push task
                # task_descriptions = ["Push red button right"]
                env_descriptions_by_name[env_name] = task_descriptions
                # assert self.task_descriptions_by_task_name.get(env_name) is None, (
                #     'Duplicate env in train and heldout envs')
                self.task_descriptions_by_task_name[env_name] = copy.deepcopy(task_descriptions)

            if self.language_emb_model:
                self.language_emb_model.cache_task_embedding_for_tasks(env_descriptions_by_name)
                print(f'Did create task embeddings for {env_type}: {len(env_descriptions_by_name)}')

    def get_action(self, observation_dict: Dict):
        expanded_observation_dict = {}
        for k, v in observation_dict.items():
            if k == 'task_idx':
                expanded_observation_dict[k] = torch.LongTensor([v])
            elif isinstance(v, np.ndarray):
                expanded_observation_dict[k] = np.expand_dims(v, 0)
            elif isinstance(v, str):   # task
                expanded_observation_dict[k] = [v]
            elif isinstance(v, list):   # all_tasks
                expanded_observation_dict[k] = [v]
            elif k == 'gt_state' and isinstance(v, dict):
                # This is gt_state for envs that have dict observations e.g. multiobj multiskill
                pass
            else:
                breakpoint()
                raise ValueError(f"Invalid observation key: {k}")

            if k == 'task_enc':
                # one-hot observation for now.
                expanded_observation_dict[k] = torch.FloatTensor(
                        expanded_observation_dict[k]).to(self.device)

        with torch.no_grad():
            action = self.run_on_batch(expanded_observation_dict, use_eval_transforms=True)
         
        if isinstance(action, dict):
            assert self.policy.use_GMM_action, "Only GMM policies should return dict"
            # For GMM policies (or more complex) policies which require more processing
            # simply forward to policy
            sampled_action, action_info = self.policy.get_noisy_action(action)
            return sampled_action, action_info

        else:
            # TODO(Mohit): Calculate mu + eps for action sampling
            action = action.cpu().numpy().ravel()
            return self.policy.get_noisy_action(action)

    def run_on_multiview_batch(self, observation_dict: Dict, use_eval_transforms: bool = False,
                               save_augs_and_return_cfg: Optional[Mapping[str, Any]] = None,
                               return_info_dict: bool = False,):
        processed_images = dict()
        current_tasks = observation_dict['task']

        # if 'peg_insert_circular_var_0' not in self.task_descriptions_by_task_name:
        #     self.task_descriptions_by_task_name['peg_insert_circular_var_0'] = (
        #         self.task_descriptions_by_task_name['RealworldBlockInsert_var_0'])

        # Take the first description for now for each task.
        if self.randomize_task_description_sampling:
            task_descriptions = [np.random.choice(self.task_descriptions_by_task_name[tn]) for tn in current_tasks]
        else:
            task_descriptions = [self.task_descriptions_by_task_name[tn][0] for tn in current_tasks]

        task_embs = None
        if self.language_emb_model:
            task_embs = self.language_emb_model.forward(
                current_tasks,
                randomize_task_emb_sampling=self.randomize_task_description_sampling)

        proprio_inp = torch.Tensor(observation_dict['proprio']).float().to(self.device)
        proprio_z = self.proprio_encoder(proprio_inp)
        observation_dict['proprio_z'] = proprio_z
        observation_dict['task_emb'] = task_embs
        observation_dict['task_descriptions'] = task_descriptions

        out, info_dict = self.image_encoder.forward_all_cameras(
            self.camera_names,
            tasks=current_tasks,
            task_embs=task_embs,
            task_descriptions=task_descriptions,
            obs_info=observation_dict,
            use_eval_transforms=use_eval_transforms,
            device=self.device,
            save_augs_and_return_cfg=save_augs_and_return_cfg,
        )
        if save_augs_and_return_cfg is not None and save_augs_and_return_cfg['use']:
            return info_dict
        
        img_lang_z = out
        observation_dict['img_z'] = img_lang_z
        action_out = self.policy.forward(observation_dict, force_cpu=False)
        if return_info_dict:
            return action_out, info_dict
        else:
            return action_out

    def run_on_batch(self, observation_dict: Dict, use_eval_transforms: bool = False,
                     save_augs_and_return_cfg: Optional[Mapping[str, Any]] = None,
                     return_info_dict: bool = False):

        if (len(self.camera_names) > 1 and hasattr(self.image_encoder, 'use_multiview_batch')
            and self.image_encoder.use_multiview_batch):
            return self.run_on_multiview_batch(observation_dict, use_eval_transforms=use_eval_transforms,
                                               save_augs_and_return_cfg=save_augs_and_return_cfg,
                                               return_info_dict=return_info_dict)

        assert save_augs_and_return_cfg is None, "Not used for single view for now"
        processed_images = dict()

        if isinstance(self.image_encoder.transforms, dict):
            transforms = None
            transform_type = 'eval' if use_eval_transforms else 'train'
        else:
            transforms = (self.image_encoder.eval_transforms
                if use_eval_transforms else self.image_encoder.transforms)
        
        current_tasks = observation_dict['task']

        # Take the first description for now for each task.
        if self.randomize_task_description_sampling:
            task_descriptions = [np.random.choice(self.task_descriptions_by_task_name[tn]) for tn in current_tasks]
        else:
            task_descriptions = [self.task_descriptions_by_task_name[tn][0] for tn in current_tasks]

        task_embs = None
        if self.language_emb_model is not None:
            task_embs = self.language_emb_model.forward(
                current_tasks,
                randomize_task_emb_sampling=self.randomize_task_description_sampling)

        proprio_inp = torch.Tensor(observation_dict['proprio']).float().to(self.device)
        proprio_z = self.proprio_encoder(proprio_inp)
        observation_dict['proprio_z'] = proprio_z
        observation_dict['task_emb'] = task_embs
        observation_dict['task_descriptions'] = task_descriptions

        # assert len(self.camera_names) == 1
        for image_key in self.camera_names:
            processed_images[image_key] = []

            for o in observation_dict[image_key]:
                if torch.is_tensor(o):
                    if transforms is not None:
                        try:
                            o_t = torch.permute(o/255., (2, 0, 1))
                            img = transforms(o_t).unsqueeze(0)
                        except:
                            img = transforms(Image.fromarray(o.numpy().astype(np.uint8))).unsqueeze(0)
                    else:
                        img = self.image_encoder.transforms[image_key][transform_type](Image.fromarray(o.numpy().astype(np.uint8))).unsqueeze(0)
                else:
                    if transforms is not None:
                        try:
                            o_t = torch.permute(torch.from_numpy(o/255.).to(torch.float32), (2, 0, 1))
                            img = transforms(o_t).unsqueeze(0)
                        except:
                            img = transforms(Image.fromarray(o.astype(np.uint8))).unsqueeze(0)
                    else:
                        img = self.image_encoder.transforms[image_key][transform_type](Image.fromarray(o.astype(np.uint8))).unsqueeze(0)
                
                processed_images[image_key].append(img)

            imgs = torch.cat(processed_images[image_key]).to(self.device)
            if hasattr(self.image_encoder, 'output_task_how_encoding') and self.image_encoder.output_task_how_encoding:
                imgs_z, task_how_embd = self.image_encoder(imgs, tasks=current_tasks, task_embs=task_embs,
                                            task_descriptions=task_descriptions,
                                            obs_info=observation_dict,)
                observation_dict[image_key] = imgs_z
                observation_dict['task_how_embd'] = task_how_embd
            else:
                imgs_z = self.image_encoder(imgs, tasks=current_tasks, task_embs=task_embs,
                                            task_descriptions=task_descriptions,
                                            obs_info=observation_dict,camera_name=image_key)
                observation_dict[image_key] = imgs_z

        # proprio_inp = torch.Tensor(observation_dict['proprio']).float().to(self.device)
        # proprio_z = self.proprio_encoder(proprio_inp)

        # Update observations before passing them to the actor (policy)
        if len(self.camera_names) == 1:
            observation_dict['img_z'] = imgs_z
        else:
            observation_dict['img_z'] = self.image_encoder.encode_multiple_views(observation_dict,self.camera_names)

        action_out = self.policy.forward(observation_dict, force_cpu=False)
        del imgs
        return action_out

    def train(self):
        """Set model state to train."""
        if self.freeze_image_encoder:
            self.image_encoder.eval()
        else:
            self.image_encoder.train()
        self.proprio_encoder.train()
        self.policy.model.train()

    def eval(self):
        """Set model state to eval."""
        self.image_encoder.eval()
        self.proprio_encoder.eval()
        self.policy.model.eval()

    def get_state_dict(self):
        self.send_to_device(torch.device('cpu'))
        state_dict = {
            'image_encoder': self.image_encoder.state_dict(),
            'proprio_encoder': self.proprio_encoder.state_dict(),
        }
        state_dict['policy'] = self.policy.get_state_dict()
        self.send_to_device(self.device)
        return state_dict

    def load_state_dict(self, state_dict):
        self.send_to_device(torch.device('cpu'))
        self.image_encoder.load_state_dict(state_dict['image_encoder'])
        self.proprio_encoder.load_state_dict(state_dict['proprio_encoder'])
        self.policy.load_state_dict(state_dict['policy'])
        self.send_to_device(self.device)
