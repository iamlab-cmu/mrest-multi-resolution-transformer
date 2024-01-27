import hydra
from tabulate import tabulate
import gym, wandb
import numpy as np, time as timer, multiprocessing, pickle, os
import os
from typing import Any, Dict, List, Mapping, Optional, Union
from pathlib import Path
import gc

import random
import socket
import time
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from mrest.utils.sampling import sample_paths
from mrest.utils.gaussian_mlp import MLP
from mrest.utils.behavior_cloning_with_encoder_multi_task import BCWithEncoderMultiTask
from mrest.utils.vision_models.resnet_with_film_encoder import ResnetFilmImageEncoder

from mrest.utils.vision_models.mdetr_multiview_policy import MDETRMultiViewPolicy
from mrest.utils.vision_models.mdetr_multiview_concat_policy import MDETRMultiViewBaselinePolicy

from mrest.utils.encoder_with_proprio import EncoderWithProprio, ProprioEncoder
from mrest.utils.bc_datasets import BCMultiTaskVariableLenImageProprioDataset, BCPybulletMultiTaskVariableLenImageProprioDataset
from mrest.utils.bc_datasets_realworld import RealWorldPegInsertDataset
from mrest.utils.logger import DataLog, TopKLogger, convert_config_dict_to_dict

from mrest.utils.env_helpers import (create_envs_from_multitask_config, read_config_for_parameterized_envs,
                                       create_single_parameterized_env_with_name, filter_train_val_task_names,
                                       create_single_pybullet_parameterized_env_with_name)
from mrest.utils.language.task_language_embedding_model import TaskEmbeddingController

from mrest.utils.env_helpers import RealWorldEnvVariationData, parse_realworld_env_configs_to_env_names
from mrest.utils.env_helpers_realworld import create_realworld_env_with_name
from mrest.utils.torch_utils import check_config_flag

if os.environ.get('USE_RLBENCH') == '1':
    from mrest.utils.env_helpers_rlbench import create_rlbench_env_with_name
    from mrest.utils.bc_datasets_rlbench import RLBenchDataset
    from mrest.utils.env_helpers import parse_rlbench_env_configs_to_env_names, RLBenchEnvVariationData

from moviepy.editor import ImageSequenceClip
from mrest.utils.mdetr.mdetr_object_detection import load_model_mdetr


def make_agent(env_specs,
               env_kwargs: dict,
               bc_kwargs: dict,
               proprio_encoder_kwargs: dict,
               policy_config: dict,
               policy_encoder_kwargs: dict,
               image_encoder_kwargs: dict,
               language_config: dict,
               resnet_film_config: dict,
               mdetr_config: dict,
               train_dl: Optional[Any] = None,
               val_dl: Optional[Any] = None,
               dl_info: Optional[Mapping[str, Any]] = None,
               epochs: int = 1,
               make_bc_agent: bool = True,
               camera_names = ['left_cap2'],
               job_config: Optional[Mapping[str, Any]] = None,) -> Union[EncoderWithProprio, BCWithEncoderMultiTask]:

    # Create model
    embedding_name = env_kwargs['embedding_name']
    common_kwargs = image_encoder_kwargs.get('common', {})

    if 'resnet' in embedding_name and 'film' in embedding_name:
        image_encoder = ResnetFilmImageEncoder(
            env_kwargs['embedding_name'],
            env_kwargs['load_path'],
            resnet_film_config,
            **common_kwargs,
            **image_encoder_kwargs.get(embedding_name, {}),
        )
    elif 'mdetr' in embedding_name:
        if mdetr_config.use_proprio:
            if policy_config['type'] == 'shared':
                assert not policy_config['shared']['concat_proprio'], "proprio being used twice."
        if 'multiview_baseline' in embedding_name or 'multiview_baseline' in embedding_name:
            print("===== Using MultiView baseline policy ====")
            image_encoder = MDETRMultiViewBaselinePolicy(
                env_kwargs['embedding_name'],
                env_kwargs['load_path'],
                mdetr_config,
                **common_kwargs,
                **image_encoder_kwargs.get(embedding_name, {}),
            )
        elif 'multiview' in embedding_name or 'multiview' in embedding_name:
            image_encoder = MDETRMultiViewPolicy(
                env_kwargs['embedding_name'],
                env_kwargs['load_path'],
                mdetr_config,
                **common_kwargs,
                **image_encoder_kwargs.get(embedding_name, {}),
            )
        else:
            raise NotImplementedError

    else:
        print(f"Requested model ({env_kwargs['embedding_name']}) not available")
        raise NotImplementedError

    proprio_encoder = ProprioEncoder(
        proprio_encoder_kwargs['proprio'],
        proprio_encoder_kwargs['hidden_sizes'],
        proprio_encoder_kwargs['final_emb_size'],
        proprio_encoder_kwargs['nonlinearity'],
    )

    # Create the language config
    language_emb_model = None
    language_emb_size = 0
    if language_config is not None and language_config['use']:
        language_emb_model = TaskEmbeddingController(language_config)
        language_emb_size = language_emb_model.task_embedding_dim

    # Creates MLP (Where the FC Network has a batchnorm in front of it)
    mlp_policy_input_dim = image_encoder.output_embedding_size + proprio_encoder.output_embedding_size
    policy_type = policy_config['type']

    if policy_type == 'shared':
        # For MDETR there is an option in which we can
        concat_proprio = policy_config['shared']['concat_proprio']
        if not concat_proprio:
            # assert 'mdetr' in embedding_name and mdetr_config.use_proprio, (
            #     'Concat proprio not set and not using mdetri embedding with proprio. '
            #     'Proprio will not be used!!')
            mlp_policy_input_dim -= proprio_encoder.output_embedding_size

        concat_lang = policy_config['shared'].get('concat_language', False)
        if 'mdetr' in embedding_name and concat_lang:
            mlp_policy_input_dim += image_encoder.task_how_encoder.output_dim
        policy = MLP(env_specs, mlp_policy_input_dim, **policy_config['shared'])
        policy.model.proprio_only = False

    else:
        raise ValueError(f"Invalid policy type: {policy_type}")

    policy_with_encoder = EncoderWithProprio(
        image_encoder,
        proprio_encoder,
        policy,
        language_emb_model,
        camera_names,
        policy_encoder_kwargs['proprio_keys'],
        policy_encoder_kwargs['finetune_image_encoder'],
        bc_kwargs.optimizer,   # Only used for debugging purposes,
        randomize_task_description_sampling=policy_encoder_kwargs.get(
            'randomize_task_description_sampling', False),
        append_object_mask=common_kwargs.get('append_object_mask', None),
    )

    if make_bc_agent:
        assert train_dl is not None, 'Cannot create bc agent without dataloader.'
        bc_agent = BCWithEncoderMultiTask(
            train_dl,
            val_dl,
            policy_with_encoder=policy_with_encoder,
            epochs=epochs,
            set_transforms=False,
            camera_name=env_kwargs['camera_name'],
            **bc_kwargs)

        return bc_agent
    else:
        return policy_with_encoder


def configure_cluster_GPUs(gpu_logical_id: int) -> int:
    # get the correct GPU ID
    if "SLURM_STEP_GPUS" in os.environ.keys():
        physical_gpu_ids = os.environ.get('SLURM_STEP_GPUS')
        gpu_id = int(physical_gpu_ids.split(',')[gpu_logical_id])
        print("Found slurm-GPUS: <Physical_id:{}>".format(physical_gpu_ids))
        print("Using GPU <Physical_id:{}, Logical_id:{}>".format(gpu_id, gpu_logical_id))
    else:
        gpu_id = 0 # base case when no GPUs detected in SLURM
        print("No GPUs detected. Defaulting to 0 as the device ID")
    return gpu_id


def get_reduced_stats_for_evaluation(logger) -> Mapping[str, float]:
    final_success_log = logger.replace_values_with_key(
            'eval_success/', max, 'max_success/')
    final_success_log.update(logger.replace_values_with_key(
            'eval_any_success/', max, 'max_any_success/'))
    final_success_log.update(logger.replace_values_with_key(
            'eval_success/', np.mean, 'mean_success/'))
    final_success_log.update(logger.replace_values_with_key(
            'eval_any_success/', np.mean, 'mean_any_success/'))
    final_success_log.update(logger.replace_values_with_key(
            'eval_success/', lambda x: np.mean(sorted(x)[-5:]), 'mean_top5_success/'))
    final_success_log.update(logger.replace_values_with_key(
            'eval_any_success/', lambda x: np.mean(sorted(x)[-5:]), 'mean_top5_any_success/'))
    final_success_log.update(logger.replace_values_with_key(
            'eval_success/', lambda x: np.mean(x[-5:]), 'mean_last5_success/'))
    final_success_log.update(logger.replace_values_with_key(
            'eval_any_success/', lambda x: np.mean(x[-5:]), 'mean_last5_any_success/'))

    final_success_log.update(logger.replace_values_with_key(
            'eval_reward/', max, 'max_reward/'))
    final_success_log.update(logger.replace_values_with_key(
            'eval_reward/', np.mean, 'mean_reward/'))
    final_success_log.update(logger.replace_values_with_key(
            'eval_reward/', lambda x: np.mean(sorted(x)[-5:]), 'mean_top5_reward/'))
    final_success_log.update(logger.replace_values_with_key(
            'eval_reward/', lambda x: np.mean(x[-5:]), 'mean_last5_reward/'))
    return final_success_log


def eval_on_envs(envs: Union[List[str], List[gym.Env]],
                 policy: EncoderWithProprio,
                 logger: DataLog,
                 job_data: DictConfig,
                 record_per_env_eval: bool,
                 num_episodes: int,
                 seed_offset: int,
                 file_log_steps: int,
                 env_type = 'train',
                 delete_envs: bool = False,
                 env_config_dict: Optional[Mapping[str, Any]] = None,
                 is_env_type_train: Optional[bool] = None,
                 set_numpy_seed_per_env: Optional[bool] = True,):
    """Run policy on envs.

    Args:
      envs: List[env] envs to run policy on.
      policy: EncoderWithProprio. Policy encoder.
      logger: Logger to log results.
      jog_data: Config.
      record_per_env_eval: Record statistics for each env separately.
      num_episodes: Number of episodes to run for each env.
      seed_offset: Offset used for each env seed. Creates different env positions.
      file_log_steps: Filename offsets used to save video rollouts.
      delete_envs: if True delete envs after rollouts. Used to save GPU memory else in
          settings where we have too many environments (200), we end up using too much memory.
    """
    all_envs_success_percentage = []
    all_envs_any_success_percent = []
    all_envs_mean_reward = []

    sim_env_type = job_data.get('env_type', 'metaworld')
    env_suite_to_use = job_data[f'{sim_env_type}_envs']['use']
    env_suite_config = job_data[f'{sim_env_type}_envs'][env_suite_to_use]
    camera_names = env_suite_config.get('camera_names', ['left_cap2'])
    is_lazy_env = False

    if isinstance(envs[0], str):
        is_lazy_env = True
        if sim_env_type == 'metaworld' or sim_env_type == 'pybullet':
            assert env_config_dict is not None, 'Need env config to create lazy envs.'
    elif isinstance(envs[0], RLBenchEnvVariationData):
        is_lazy_env = True

    set_numpy_seed_per_env = False if set_numpy_seed_per_env is None else set_numpy_seed_per_env

    # TODO: Implement this for RLBench.
    append_object_mask = job_data['image_encoder_kwargs']['common'].get('append_object_mask', None)
    if append_object_mask == 'mdetr':
        obj_detection_model = load_model_mdetr()
        obj_detection_processor = None
    else:
        obj_detection_model, obj_detection_processor = None, None

    envs_iter = range(len(envs))
    for e_idx in tqdm(envs_iter) if sim_env_type == 'metaworld' else envs_iter:
        # collect_paths_start_time = time.time()
        e = envs[e_idx]
        # Lazy env creation only supported for parameterized envs
        if is_lazy_env:
            if sim_env_type == 'metaworld':
                assert isinstance(e, str), f'Invalid env type: {e}'
                e = create_single_parameterized_env_with_name(job_data, e, env_config_dict, camera_names=camera_names)
            elif sim_env_type == 'pybullet':
                e = create_single_pybullet_parameterized_env_with_name(job_data, e, env_config_dict, camera_names=camera_names)
            elif sim_env_type == 'rlbench':

                # if 'place_shape_in_shape_sorter_ms' not in e:
                #     print(f'Skipping env: {e}')
                #     continue
                if isinstance(e, str):
                    if '_data_' in e:
                        e = e.split('_data_')[0]
                    e_name, e_variation = e.split('_var_')
                    e_info = RLBenchEnvVariationData(e_name, int(e_variation), '0', None)
                else:
                    assert isinstance(e, RLBenchEnvVariationData)
                    e_name = e.env
                    e_info = e

                print(f"Will create: {e_name}")
                e = create_rlbench_env_with_name(job_data, e_info, camera_names=camera_names, headless=True)

            elif sim_env_type == 'realworld':
                assert isinstance(e, str), f'Invalid env type: {e}'
                if '_data_' in e:
                    e = e.split('_data_')[0]
                if '_var_' in e:
                    e_name, e_variation = e.split('_var_')
                else:
                    e_name, e_variation = e, '0'
                e_info = RealWorldEnvVariationData(e_name, int(e_variation), '0', None)
                print(f"Will create: {e_name}")
                e = create_realworld_env_with_name(job_data, e_info, camera_names,
                                                   is_real_world=True)


        print(f"Sampling {num_episodes} paths for env {e_idx}/{len(envs)} with name {envs[e_idx]}")
        paths = sample_paths(num_episodes,
                             env=e, #env_constructor,
                             policy=policy, eval_mode=True, horizon=e.horizon,
                             base_seed=job_data['seed'] + seed_offset + e_idx * 100,
                             num_cpu=job_data['num_cpu'],
                             env_kwargs=job_data['env_kwargs'],
                             env_has_image_embedding_wrapper=False,
                             set_numpy_seed_per_env=set_numpy_seed_per_env,
                             append_object_mask=append_object_mask,
                             obj_detection_model=obj_detection_model,
                             obj_detection_processor=obj_detection_processor,
                             env_type=sim_env_type,)
        # collect_paths_end_time = time.time()
        # print(f'collect paths duration: {collect_paths_end_time - collect_paths_start_time:.4f}')
        ## Success computation and logging for MetaWorld
        sc = []
        any_success = []
        rewards = []

        should_save_gifs = job_data['env_gif_saver']['use']
        save_gif_freq = job_data['env_gif_saver']['save_env_freq']
        for i, path in enumerate(paths):
            if path['env_infos']['success'][-1].size > 1:
                any_success.append(int(np.any(np.stack(path['env_infos']['success'][1:]))))
            else:
                any_success.append(int(np.any(path['env_infos']['success'])))
            
            sc.append(path['env_infos']['success'][-1])
            rewards.append(np.sum(path['rewards']))

            hostname = socket.gethostname()
            if (should_save_gifs and e_idx % save_gif_freq == 0 and i < 10 and job_data['pixel_based']):
                vid = path.get('images')
                def _get_video_fps(_vid):
                    if sim_env_type == 'metaworld':
                        return 20
                    else:
                        if len(_vid) < 15:
                            fps = 1
                        elif len(_vid) < 51:
                            fps = 4
                        elif len(_vid) < 100:
                            fps = 10
                        else:
                            fps = 20
                        return fps
                def _remove_time_from_video(_vid: List[np.ndarray]):
                    if _vid[-1].ndim == 3:
                        return _vid
                    elif _vid[-1].ndim == 4:
                        return [v[-1] for v in _vid]
                    else:
                        raise ValueError(f'Invalid video shape: {vid[-1].shape}')

                if (vid is not None and len(vid) == 0) and len(path['images_by_camera_name']) == 1:
                    vid, = path['images_by_camera_name'].values()
                    vid = _remove_time_from_video(vid)

                if (vid is not None and len(vid) > 0):
                    vid = _remove_time_from_video(vid)
                    filename = f'./iterations/steps_{file_log_steps:04d}/{e.env_id}/vid_{i}_succ_{any_success[-1]}.gif'
                    if not os.path.isdir(f'iterations/steps_{file_log_steps:04d}/{e.env_id}'):
                        os.makedirs(f'iterations/steps_{file_log_steps:04d}/{e.env_id}')
                    cl = ImageSequenceClip(vid, fps=_get_video_fps(vid))
                    cl.write_gif(filename, fps=_get_video_fps(vid), logger=None)
                    del cl

                # TODO: Unsure if this below code is needed or not.
                # if len(camera_names) == 1:
                    # vid = path['images_by_camera_name']['left_cap2']
                    # if vid is not None:
                        # filename = f'./iterations/steps_{file_log_steps:04d}/{e.env_id}/vid_{i}_succ_{any_success[-1]}.gif'
                        # if not os.path.isdir(f'iterations/steps_{file_log_steps:04d}/{e.env_id}'):
                            # os.makedirs(f'iterations/steps_{file_log_steps:04d}/{e.env_id}')
                        # cl = ImageSequenceClip(vid, fps=_get_video_fps(vid))
                        # cl.write_gif(filename, fps=_get_video_fps(vid), logger=None)
                        # del cl

                if len(camera_names) == 2:
                    if (camera_names[0] + '-' + camera_names[1]) in path['images_by_camera_name']:
                        cam_name = camera_names[0] + '-' + camera_names[1]
                    else:
                        cam_name = camera_names[1] + '-' + camera_names[0]
                    vid = path['images_by_camera_name'][cam_name]
                    vid = _remove_time_from_video(vid)

                    filename = f'./iterations/steps_{file_log_steps:04d}/{e.env_id}/{cam_name}_{i}_succ_{any_success[-1]}.gif'
                    if not os.path.isdir(f'iterations/steps_{file_log_steps:04d}/{e.env_id}'):
                        os.makedirs(f'iterations/steps_{file_log_steps:04d}/{e.env_id}')
                    cl = ImageSequenceClip(vid, fps=_get_video_fps(vid))
                    cl.write_gif(filename, fps=_get_video_fps(vid), logger=None)
                    del cl

        success_percentage = np.mean(sc) * 100.0
        any_success_percent = np.mean(any_success) * 100.0
        mean_reward = np.mean(rewards)

        # Logging
        if record_per_env_eval:
            logger.log_kv(f'eval_success/{env_type}/{e.env_id}', success_percentage)
            logger.log_kv(f'eval_any_success/{env_type}/{e.env_id}', any_success_percent)
            logger.log_kv(f'eval_reward/{env_type}/{e.env_id}', mean_reward)
        all_envs_success_percentage.append(success_percentage)
        all_envs_any_success_percent.append(any_success_percent)
        all_envs_mean_reward.append(mean_reward)

        if delete_envs:
            # Close the environment to release resources
            if sim_env_type == 'rlbench':
                e.close()
            # Do not delete the string
            if not is_lazy_env:
                envs[e_idx] = None
            del e
            del path
            del paths
            gc.collect()
            torch.cuda.empty_cache()
    del obj_detection_model
    del obj_detection_processor
    return all_envs_success_percentage, all_envs_any_success_percent, all_envs_mean_reward


def run_train_eval_loop(job_data, agent, envs_dict, train: bool = True, delete_envs: bool = False):

    least_loss = np.inf
    hostname = socket.gethostname()

    topk_logger = TopKLogger(job_data['wandb']['saver'].get('save_top_k', 5))

    # Log initial stufff
    # torch.cuda.empty_cache()
    # with torch.no_grad():
    #     agent.log_image_augmentation_data(num_batches=4, num_images_per_batch=8)

    for epoch in range(job_data['epochs']):
        if train:
            log_epochs, log_steps = agent.train_epochs, agent.train_steps
        else:
            log_epochs, log_steps = epoch, epoch

        # Evaluation
        if job_data['agent_eval']['use'] and log_epochs % job_data['agent_eval']['eval_freq_epochs'] == 0:
            print(f"===========Evaluating at epoch {log_epochs}")
            env_suite_to_use = job_data['metaworld_envs']['use']
            record_per_env_eval = job_data['metaworld_envs'][env_suite_to_use]['record_per_env_eval']

            agent.policy.eval()

            for env_type, envs in envs_dict.items():
                if len(envs) == 0:
                    print(f"Found no env for env type: {env_type}")
                    continue

                if (isinstance(job_data['eval_num_traj'], dict) or
                    isinstance(job_data['eval_num_traj'], DictConfig)):
                    num_episodes = job_data['eval_num_traj'][env_type]
                else:
                    num_episodes = job_data['eval_num_traj']

                (success_percent, any_success_percent, mean_reward) = eval_on_envs(
                    envs, agent.policy, agent.logger, job_data, record_per_env_eval, num_episodes,
                    log_epochs, log_steps, env_type=env_type, delete_envs=delete_envs)

                agent.logger.log_kv(f'eval_success/all_{env_type}_envs', np.mean(success_percent))
                agent.logger.log_kv(f'eval_any_success/all_{env_type}_envs', np.mean(any_success_percent))
                agent.logger.log_kv(f'eval_reward/all_{env_type}_envs', np.mean(mean_reward))
            agent.logger.log_kv('eval_epoch', log_epochs)

            # Get the set of statistics we want and log them.
            # NOTE: While we only care about final statistics we save them repeatedly so that in
            # case of any crash we can still get some relevant values.
            final_success_log = get_reduced_stats_for_evaluation(agent.logger)
            for k, v in final_success_log.items():
                agent.logger.log_kv(k, v)
            agent.logger.save_log('./logs/')
            agent.logger.save_wb(step=log_steps, filter_step_key=True)

            print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                        agent.logger.get_current_log().items()))
            print(tabulate(print_data))

            # Save policy (only useful when train = True), hence we can use agent.train_epochs
            if train and job_data['wandb']['saver']['upload']:
                print(f"===Saving policy at epoch {agent.train_epochs}===")
                # NOTE: Should this any_success be for train or eval (heldout)?
                any_success_percent_mean = np.mean(any_success_percent)
                save_file_name = (f'./iterations/'
                    f'encoder_proprio_policy_epoch{agent.train_epochs}_'
                    f'anySuccess{any_success_percent_mean:.3f}.pth')
                status = topk_logger.push(save_file_name, any_success_percent_mean)
                if status:
                    torch.save(agent.policy.get_state_dict(), save_file_name)
                    wandb.save(save_file_name, base_path=str(os.getcwd()))

        # update policy using one BC epoch
        if train:
            agent.policy.train()
            log_dict = agent.train(job_data['pixel_based'], suppress_fit_tqdm=True, step=agent.train_steps)
            print(f"Train steps: {agent.train_steps}, loss: {log_dict['loss']:.4f}")
            log_epochs, log_steps = agent.train_epochs, agent.train_steps
        else:
            log_epochs, log_steps = epoch, epoch

        # Validation
        print(f'agent epochs {log_epochs}')
        if train and ((agent.train_epochs % job_data['val_freq_epochs']) == 0) and job_data['log_val']:
            del log_dict
            torch.cuda.empty_cache()
            print(f"Validating at epoch: {agent.train_epochs}")
            agent.policy.eval()
            with torch.no_grad():
                val_log_dict = agent.run_on_validation_data(job_data['pixel_based'], suppress_fit_tqdm=True, step=agent.train_steps)
            print(f"==== Train steps: {agent.train_steps}, val_loss: {val_log_dict['val_loss']:.4f} ===")

            # if (val_log_dict['val_loss'] < least_loss) and job_data['save_best_val_policy']:
            #     least_loss = val_log_dict['val_loss'] + 0.
            #     pickle.dump(agent.policy, open(f'./iterations/best_val_policy_epoch{epoch}.pickle', 'wb'))
            #     torch.save(agent.policy.get_state_dict(), f'./iterations/best_val_encoder_proprio_policy_epoch{epoch}.pth')

        # Save periodically (if enabled, only useful if train = True)
        if train and (job_data['wandb']['saver_no_eval'].get('use', False) and
            agent.train_epochs % job_data['wandb']['saver_no_eval']['save_freq_epochs'] == 0):
            save_file_name = f'./checkpoints/ckpt_{agent.train_epochs:04d}.pth'
            torch.save(agent.policy.get_state_dict(), save_file_name)
            wandb.save(save_file_name, base_path=str(os.getcwd()))
            print(f'Did save checkpoint: {os.path.join(os.getcwd(), save_file_name)}')

        log_image_augmentations = False
        if (train and ((agent.train_epochs % job_data['val_freq_epochs']) == 0) and job_data['log_val'] and
            log_image_augmentations):
            torch.cuda.empty_cache()
            with torch.no_grad():
                agent.log_image_augmentation_data(num_batches=2, num_images_per_batch=8)


    # Get the final set of statistics we want and log them.
    final_success_log = get_reduced_stats_for_evaluation(agent.logger)
    for k, v in final_success_log.items():
        agent.logger.log_kv(k, v)
    agent.logger.save_wb(step=log_steps, filter_step_key=True)

    return agent


def create_realworld_dataloaders(job_data, use_train_data: bool = True,
                                 filter_cfg = None,):
    '''Create dataloaders for real-world tasks

    RealWorld tasks include:
        - Green object insertion on circular peg board.
    '''
    env_suite_to_use = job_data['realworld_envs']['use']
    job_data['envs']['type'] = job_data['realworld_envs'][env_suite_to_use]['type']
    job_data['env_kwargs']['episode_len'] = job_data['realworld_envs'][env_suite_to_use]['episode_len']
    job_data['data_dir'] = job_data['realworld_envs'][env_suite_to_use]['common_data_dir']

    env_config_dict_by_type = parse_realworld_env_configs_to_env_names(job_data, filter_cfg=filter_cfg)

    # job_data['envs']['names'] = ['RealworldBlockInsert', 'insert_peg', 'peg_insert_cicular']
    job_data['envs']['names'] = [env_info.name for env_info in env_config_dict_by_type['train_env_info']]
    job_data['envs']['num_envs'] = len(job_data['envs']['names'])
    # job_data['envs']['heldout_env_names'] = ['RealworldBlockInsert', 'insert_peg', 'peg_insert_circular']
    job_data['envs']['heldout_env_names'] = [env_info.name for env_info in env_config_dict_by_type['heldout_env_info']]
    info = {}

    print('==== Train envs ====')
    print(job_data['envs']['names'])

    train_set = RealWorldPegInsertDataset(
        job_data,
        task_info=env_config_dict_by_type['train_env_info'],
        camera_names=job_data['realworld_envs'][env_suite_to_use]['camera_names'],
        proprio_len=job_data['proprio'],
        data_subsampling_cfg=job_data['realworld_envs'].get('data_subsampling_cfg',  None),
        image_crop_cfg=job_data['realworld_envs'].get('image_crop_cfg', None),
        )

    # By reference (since we update YAML below)
    normalize_action_cfg = job_data['realworld_envs'][env_suite_to_use]['normalize_actions']
    if (normalize_action_cfg and normalize_action_cfg['use']):
        if normalize_action_cfg['values_set']:
            action_stats = normalize_action_cfg['values']
        else:
            action_stats = train_set.get_action_stats_for_dataset()
            for stat_key, stat_val in action_stats.items():
                if isinstance(stat_val, np.ndarray):
                    normalize_action_cfg['values'][stat_key] = ListConfig(stat_val.tolist())
                else:
                    normalize_action_cfg['values'][stat_key] = stat_val
            normalize_action_cfg['values_set'] = True

    val_set = RealWorldPegInsertDataset(
        job_data,
        task_info=env_config_dict_by_type['heldout_env_info'],
        camera_names=job_data['realworld_envs'][env_suite_to_use]['camera_names'],
        proprio_len=job_data['proprio'],
        data_subsampling_cfg=job_data['realworld_envs'].get('data_subsampling_cfg',  None),
        image_crop_cfg=job_data['realworld_envs'].get('image_crop_cfg', None),
        is_val_data=True,
        )

    print(f"Total demos train: {len(train_set)}, val: {len(val_set)}")
    train_iters = (len(train_set) * job_data['epochs']) // job_data['bc_kwargs']['batch_size']
    print(f"Total train iters: {train_iters}")

    info['train_task_names'] = job_data['envs']['names']

    # Combine train envs and heldout envs to creat the entire set of eval envs
    # Set max iters for annealing lr over training period.
    job_data['bc_kwargs']['scheduler']['CosineAnnealingLR']['t_max'] = train_iters

    collate_fn = None
    train_dl = DataLoader(train_set, batch_size=job_data['bc_kwargs']['batch_size'],
                          shuffle=True, num_workers=job_data['dataL_num_workers'],
                          collate_fn=collate_fn,)

    val_dl = DataLoader(val_set, job_data['bc_kwargs']['val_batch_size'],
                        shuffle=False, num_workers=job_data['dataL_num_workers'],
                        collate_fn=collate_fn,)

    return train_dl, val_dl, info


def create_rlbench_dataloaders(job_data, use_train_data: bool = True,
                               filter_cfg = None,):
    '''Create dataloaders for RLBench tasks.'''
    env_suite_to_use = job_data['rlbench_envs']['use']

    job_data['envs']['type'] = job_data['rlbench_envs'][env_suite_to_use]['type']
    job_data['env_kwargs']['episode_len'] = job_data['rlbench_envs'][env_suite_to_use]['episode_len']
    job_data['data_dir'] = job_data['rlbench_envs'][env_suite_to_use]['common_data_dir']

    env_config_dict_by_type = parse_rlbench_env_configs_to_env_names(
        job_data, filter_cfg=filter_cfg)

    job_data['envs']['names'] = [env_info.name for env_info in env_config_dict_by_type['train_env_info']]
    job_data['envs']['num_envs'] = len(job_data['envs']['names'])
    job_data['envs']['heldout_env_names'] = [env_info.name for env_info in env_config_dict_by_type['heldout_env_info']]
    info = {}

    print('==== Train envs ====')
    print(job_data['envs']['names'])

    train_set = RLBenchDataset(
        job_data,
        task_info=env_config_dict_by_type['train_env_info'],
        camera_names=job_data['rlbench_envs'][env_suite_to_use]['camera_names'],
        proprio_len=job_data['proprio'],
        data_subsampling_cfg=job_data['rlbench_envs'].get('data_subsampling_cfg',  None),
        )

    val_set = RLBenchDataset(
        job_data,
        task_info=env_config_dict_by_type['heldout_env_info'],
        camera_names=job_data['rlbench_envs'][env_suite_to_use]['camera_names'],
        proprio_len=job_data['proprio'],
        data_subsampling_cfg=job_data['rlbench_envs'].get('data_subsampling_cfg',  None),
        )

    print(f"Total demos train: {len(train_set)}, val: {len(val_set)}")
    train_iters = (len(train_set) * job_data['epochs']) // job_data['bc_kwargs']['batch_size']
    print(f"Total train iters: {train_iters}")

    info['train_task_names'] = job_data['envs']['names']

    # Combine train envs and heldout envs to creat the entire set of eval envs
    # Set max iters for annealing lr over training period.
    job_data['bc_kwargs']['scheduler']['CosineAnnealingLR']['t_max'] = train_iters

    collate_fn = None
    train_dl = DataLoader(train_set, batch_size=job_data['bc_kwargs']['batch_size'],
                          shuffle=True, num_workers=job_data['dataL_num_workers'],
                          collate_fn=collate_fn,)

    val_dl = DataLoader(val_set, batch_size=job_data['bc_kwargs']['batch_size'],
                        shuffle=False, num_workers=job_data['dataL_num_workers'],
                        collate_fn=collate_fn,)

    return train_dl, val_dl, info

def create_pybullet_dataloaders(job_data, use_train_data: bool = True,
                       max_train_tasks: Optional[int] = None,
                       percent_val_tasks: float = 0.05,
                       min_val_tasks: int = 1,
                       max_val_tasks: int = 10,
                       env_type: str = 'train',
                       filter_cfg = None):
    '''Create dataloaders

    env_type:
    '''

    # ==== Set envs ====
    env_suite_to_use = job_data['pybullet_envs']['use']
    is_parameterized_env = job_data['pybullet_envs'][env_suite_to_use]['is_parameterized']
    job_data['envs']['type'] = job_data['pybullet_envs'][env_suite_to_use]['type']
    job_data['env_kwargs']['episode_len'] = job_data['pybullet_envs'][env_suite_to_use]['episode_len']
    job_data['data_dir'] = job_data['pybullet_envs'][env_suite_to_use]['data_dir']

    info = {}

    env_suite_config = job_data['pybullet_envs'][env_suite_to_use]

    # NOTE: We should probably not be using eval_configs at all since our heldout envs are still train
    # envs and not the true eval envs.
    env_config_dict_by_type = read_config_for_parameterized_envs(
        env_suite_config.data_dir, read_all_configs=True)

    env_config_dict = env_config_dict_by_type[env_type]

    max_train_tasks = (max_train_tasks if max_train_tasks
                        else job_data['pybullet_envs'][env_suite_to_use]['max_train_tasks'])

    if ('franka' in env_suite_to_use or 'sawyer' in env_suite_to_use):
        min_val_tasks, percent_val_tasks = 0, 0 # very few envs to have any val heldout

    train_task_names = filter_train_val_task_names(env_config_dict, filter_cfg)

    job_data['envs']['names'] = train_task_names
    job_data['envs']['num_envs'] = len(job_data['envs']['names'])
    job_data['envs']['heldout_env_names'] = job_data['pybullet_envs'][env_suite_to_use]['heldout_env_names']
    job_data['envs']['heldout_env_names'] = []

    print('==== Train envs ====')
    print(train_task_names)

    num_train_demos_per_task = job_data['pybullet_envs'][env_suite_to_use].get(
        'num_demos_train_per_task', None)
    if num_train_demos_per_task is None:
        num_train_demos_per_task = job_data['num_demos_train_per_task']

    if use_train_data:
        assert env_type == 'train', f'Invalid env type for train: {env_type}'

    data_dir = env_type
    append_object_mask = job_data['image_encoder_kwargs']['common'].get('append_object_mask', None)
    if append_object_mask == 'None':
        job_data['image_encoder_kwargs']['common']['append_object_mask'] = None
        append_object_mask  = None
    
    multi_temporal_sensors = job_data.get('multi_temporal_sensors', None)

    first_key = list(env_config_dict.keys())[0] # Assuming all envs will have data saved at same frequencies
    I3_save_freq = env_config_dict[first_key]['I3_save_freq']
    Ih_save_freq = env_config_dict[first_key]['Ih_save_freq']
    FT_save_freq = env_config_dict[first_key]['FT_save_freq']
    if multi_temporal_sensors is not None:
        multi_temporal_sensors.i3_freq = int(I3_save_freq/FT_save_freq)
        multi_temporal_sensors.ih_freq = int(Ih_save_freq/FT_save_freq)
    
    train_set = BCPybulletMultiTaskVariableLenImageProprioDataset(
        data_dir=os.path.join(job_data['data_dir'], data_dir),
        task_names=train_task_names,
        num_demos_per_task=num_train_demos_per_task,
        start_idx=0,
        camera_names=env_suite_config.get('camera_names', ['left_cap2']),
        proprio_len=job_data['proprio'],
        append_object_mask=append_object_mask,
        multi_temporal_sensors=job_data.get('multi_temporal_sensors', None),
        min_action=job_data['pybullet_envs'][env_suite_to_use].get('min_action', None),
        max_action=job_data['pybullet_envs'][env_suite_to_use].get('max_action', None),
        action_type=job_data['pybullet_envs'][env_suite_to_use].get('action_type', 'delta_obs_pos'),
        )

    num_demos_val_per_task = job_data['pybullet_envs'][env_suite_to_use].get(
        'num_demos_val_per_task', None)
    if num_demos_val_per_task is None:
        num_demos_val_per_task= job_data['num_demos_val_per_task']
    val_set = BCPybulletMultiTaskVariableLenImageProprioDataset(
        data_dir=os.path.join(job_data['data_dir'], data_dir),
        task_names=train_task_names,
        num_demos_per_task=num_demos_val_per_task,
        start_idx=num_train_demos_per_task,
        camera_names=env_suite_config.get('camera_names', ['left_cap2']),
        proprio_len=job_data['proprio'],
        append_object_mask=append_object_mask,
        multi_temporal_sensors=job_data.get('multi_temporal_sensors', None),
        min_action=job_data['pybullet_envs'][env_suite_to_use].get('min_action', None),
        max_action=job_data['pybullet_envs'][env_suite_to_use].get('max_action', None),
        action_type=job_data['pybullet_envs'][env_suite_to_use].get('action_type', 'delta_obs_pos'),
        )

    print(f"Total demos train: {len(train_set)}, val: {len(val_set)}")
    train_iters = (len(train_set) * job_data['epochs']) // job_data['bc_kwargs']['batch_size']
    print(f"Total train iters: {train_iters}")

    info['train_task_names'] = train_task_names

    # Combine train envs and heldout envs to creat the entire set of eval envs
    # Set max iters for annealing lr over training period.
    job_data['bc_kwargs']['scheduler']['CosineAnnealingLR']['t_max'] = train_iters

    collate_fn = None
    train_dl = DataLoader(train_set, batch_size=job_data['bc_kwargs']['batch_size'],
                          shuffle=True, num_workers=job_data['dataL_num_workers'],
                          collate_fn=collate_fn,)

    val_dl = DataLoader(val_set, batch_size=job_data['bc_kwargs']['batch_size'],
                        shuffle=False, num_workers=job_data['dataL_num_workers'],
                        collate_fn=collate_fn,)
    return train_dl, val_dl, info

def create_dataloaders(job_data, use_train_data: bool = True,
                       max_train_tasks: Optional[int] = None,
                       percent_val_tasks: float = 0.05,
                       min_val_tasks: int = 1,
                       max_val_tasks: int = 10,
                       env_type: str = 'train',
                       filter_cfg = None):
    '''Create dataloaders

    env_type:
    '''

    # ==== Set envs ====
    env_suite_to_use = job_data['metaworld_envs']['use']
    is_parameterized_env = job_data['metaworld_envs'][env_suite_to_use]['is_parameterized']
    job_data['envs']['type'] = job_data['metaworld_envs'][env_suite_to_use]['type']
    job_data['env_kwargs']['episode_len'] = job_data['metaworld_envs'][env_suite_to_use]['episode_len']
    job_data['data_dir'] = job_data['metaworld_envs'][env_suite_to_use]['data_dir']

    info = {}
    env_suite_config = job_data['metaworld_envs'][env_suite_to_use]

    # NOTE: We should probably not be using eval_configs at all since our heldout envs are still train
    # envs and not the true eval envs.
    env_config_dict_by_type = read_config_for_parameterized_envs(
        env_suite_config.data_dir, read_all_configs=True)

    env_config_dict = env_config_dict_by_type[env_type]

    max_train_tasks = (max_train_tasks if max_train_tasks
                        else job_data['metaworld_envs'][env_suite_to_use]['max_train_tasks'])

    if ('franka' in env_suite_to_use or 'sawyer' in env_suite_to_use):
        min_val_tasks, percent_val_tasks = 0, 0 # very few envs to have any val heldout

    # ==== Example filter config ====
    # filter_cfg = OmegaConf.create({
    #         'use': True,
    #         'target_objs': ['block_red'],
    #         'skills': ['pick'],
    #         'max_envs_per_skill_target_obj_pair': 20,
    #     })

    train_task_names = filter_train_val_task_names(env_config_dict, filter_cfg)

    job_data['envs']['names'] = train_task_names
    job_data['envs']['num_envs'] = len(job_data['envs']['names'])
    job_data['envs']['heldout_env_names'] = job_data['metaworld_envs'][env_suite_to_use]['heldout_env_names']
    job_data['envs']['heldout_env_names'] = []

    print('==== Train envs ====')
    print(train_task_names)

    num_train_demos_per_task = job_data['metaworld_envs'][env_suite_to_use].get(
        'num_demos_train_per_task', None)
    if num_train_demos_per_task is None:
        num_train_demos_per_task = job_data['num_demos_train_per_task']

    if use_train_data:
        assert env_type == 'train', f'Invalid env type for train: {env_type}'

    data_dir = env_type
    append_object_mask = job_data['image_encoder_kwargs']['common'].get('append_object_mask', None)
    if append_object_mask == 'None':
        job_data['image_encoder_kwargs']['common']['append_object_mask'] = None
        append_object_mask  = None
    train_set = BCMultiTaskVariableLenImageProprioDataset(
        data_dir=os.path.join(job_data['data_dir'], data_dir),
        task_names=train_task_names,
        num_demos_per_task=num_train_demos_per_task,
        num_demos_train_per_task_cfg=job_data['metaworld_envs'][env_suite_to_use].get('num_demos_train_per_task_cfg'),
        start_idx=0,
        camera_names=env_suite_config.get('camera_names', ['left_cap2']),
        proprio_len=job_data['proprio'],
        append_object_mask=append_object_mask,
        multi_temporal_sensors=job_data.get('multi_temporal_sensors', None),
        env_config_dict=env_config_dict,
        job_data=job_data,
        )

    num_demos_val_per_task = job_data['metaworld_envs'][env_suite_to_use].get(
        'num_demos_val_per_task', None)
    if num_demos_val_per_task is None:
        num_demos_val_per_task= job_data['num_demos_val_per_task']
    val_set = BCMultiTaskVariableLenImageProprioDataset(
        data_dir=os.path.join(job_data['data_dir'], data_dir),
        task_names=train_task_names,
        num_demos_per_task=num_demos_val_per_task,
        start_idx=num_train_demos_per_task,
        camera_names=env_suite_config.get('camera_names', ['left_cap2']),
        proprio_len=job_data['proprio'],
        append_object_mask=append_object_mask,
        multi_temporal_sensors=job_data.get('multi_temporal_sensors', None),
        job_data=job_data,
        )

    print(f"Total demos train: {len(train_set)}, val: {len(val_set)}")
    train_iters = (len(train_set) * job_data['epochs']) // job_data['bc_kwargs']['batch_size']
    print(f"Total train iters: {train_iters}")

    info['train_task_names'] = train_task_names

    # Combine train envs and heldout envs to creat the entire set of eval envs
    # Set max iters for annealing lr over training period.
    job_data['bc_kwargs']['scheduler']['CosineAnnealingLR']['t_max'] = train_iters

    collate_fn = None
    train_dl = DataLoader(train_set, batch_size=job_data['bc_kwargs']['batch_size'],
                          shuffle=True, num_workers=job_data['dataL_num_workers'],
                          collate_fn=collate_fn,)

    val_dl = DataLoader(val_set, batch_size=job_data['bc_kwargs']['batch_size'],
                        shuffle=False, num_workers=job_data['dataL_num_workers'],
                        collate_fn=collate_fn,)

    return train_dl, val_dl, info


def bc_train_multitask(job_data:dict) -> None:

    # configure GPUs
    os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    physical_gpu_id = 2 #configure_cluster_GPUs(job_data['env_kwargs']['render_gpu_id'])
    # physical_gpu_id = np.random.randint(3)
    # job_data['env_kwargs']['render_gpu_id'] = physical_gpu_id

    assert (job_data.wandb.saver_no_eval.use ^ job_data.agent_eval.use), (
        'Not evaluating or saving any trained ckpts.')

    if job_data.bc_kwargs.loss_type == 'mse_binary_gripper':
        assert job_data.policy_config[job_data.policy_config.type]['binary_gripper_config']['use'], (
            'Using mse_binary_gripper as loss but not setting binary gripper in policy.'
        )
    if (job_data.policy_config[job_data.policy_config.type] is not None and
        'binary_gripper_config' in job_data.policy_config[job_data.policy_config.type] and
        job_data.policy_config[job_data.policy_config.type].binary_gripper_config.use):
        assert job_data.bc_kwargs.loss_type == 'mse_binary_gripper', (
            'Setting binary_gripper in policy but not using it in loss')

    # ==== Set seed ====
    seed = job_data['seed']
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Creating dataset
    if job_data['dataL_num_workers'] > 0:
        torch.set_num_threads(job_data['dataL_num_workers'])

    env_type = job_data.get('env_type', 'metaworld')
    if env_type == 'metaworld':
        train_dl, val_dl, dl_info = create_dataloaders(job_data, use_train_data=True,)
    elif env_type == 'rlbench':
        train_dl, val_dl, dl_info = create_rlbench_dataloaders(job_data, use_train_data=True,)
    elif env_type == 'realworld':
        train_dl, val_dl, dl_info = create_realworld_dataloaders(job_data, use_train_data=True,)
    elif  env_type == 'pybullet':
        train_dl, val_dl, dl_info = create_pybullet_dataloaders(job_data, use_train_data=True,)
    else:
        raise ValueError(f'Invalid env type: {env_type}')

    job_data['envs']['names'] = dl_info['train_task_names']
    job_data['envs']['num_envs'] = len(job_data['envs']['names'])
    job_data['envs']['heldout_env_names'] = []

    # Make log dir
    if os.path.isdir(job_data['job_name']) == False:
        os.mkdir(job_data['job_name'])

    hydra_dir = os.getcwd()
    os.chdir(job_data['job_name']) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False:
        os.mkdir('iterations')
    if os.path.isdir('logs') == False:
        os.mkdir('logs')
    # Stores all checkpoints (irrespective of eval metrics)
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    env_type = job_data.get('env_type', 'metaworld')

    if env_type == 'metaworld':
        env_suite_to_use = job_data['metaworld_envs']['use']
        env_suite_config = job_data['metaworld_envs'][env_suite_to_use]
    elif env_type == 'rlbench':
        env_suite_to_use = job_data['rlbench_envs']['use']
        env_suite_config = job_data['rlbench_envs'][env_suite_to_use]
    elif env_type == 'realworld':
        env_suite_to_use = job_data['realworld_envs']['use']
        env_suite_config = job_data['realworld_envs'][env_suite_to_use]
    elif env_type == 'pybullet':
        env_suite_to_use = job_data['pybullet_envs']['use']
        env_suite_config = job_data['pybullet_envs'][env_suite_to_use]

    camera_names = env_suite_config.get('camera_names', ['left_cap2'])

    ## Create agent and environments
    # NOTE: Assess env_name by calling env.env_id
    lazy_env_creation = True
    if lazy_env_creation:
        if env_type == 'metaworld' or env_type == 'rlbench':
            train_env_names_dict = create_envs_from_multitask_config(job_data, return_only_env_names=True)
    else:
        train_envs, heldout_envs = create_envs_from_multitask_config(job_data)

    # Create a callable to lazily create envs as required
    def metaworld_env_callable(env_name: str, is_env_type_train: bool):
        del is_env_type_train
        return create_single_parameterized_env_with_name(
            job_data, env_name, train_env_names_dict['env_config_dict'], camera_names=camera_names)

    def rlbench_env_callable(task_info, is_env_type_train: bool):
        rlbench_camera_names = [x[:-4] if x.endswith('_rgb') else x for x in camera_names]
        del is_env_type_train
        return create_rlbench_env_with_name(job_data, task_info, rlbench_camera_names)

    def realworld_env_callable(task_info, is_env_type_train: bool):
        del is_env_type_train
        return create_realworld_env_with_name(job_data, task_info, camera_names)

    def env_callable(*args, **kwargs):
        if env_type == 'metaworld':
            return metaworld_env_callable(*args, **kwargs)
        elif env_type == 'rlbench':
            return rlbench_env_callable(*args, **kwargs)
        elif env_type == 'realworld':
            return realworld_env_callable(*args, **kwargs)
        else:
            return None

    if lazy_env_creation:
        if env_type == 'metaworld':
            env_spec = env_callable(train_env_names_dict['train_env_names'][0], True).spec
        elif env_type == 'rlbench':
            env = env_callable(train_env_names_dict['train_env_info'][0], True)
            env_spec = env.spec
            env.close()
        elif env_type == 'realworld':
            env = env_callable(train_env_names_dict['train_env_info'][0], True)
            env_spec = env.spec
        elif env_type == 'pybullet':
            # TODO(saumya): remove hardcoded
            env_spec = OmegaConf.create({"observation_dim":4, "action_dim": 4})
    else:
        env_spec = train_envs[0].spec
        del heldout_envs
        heldout_envs = []
        envs_dict = {'train': train_envs}

    env_kwargs = job_data['env_kwargs']


    agent = make_agent(
        env_spec, # all envs should have same spec (observation_dim, action_dim)
        env_kwargs=env_kwargs,
        bc_kwargs=job_data['bc_kwargs'],
        proprio_encoder_kwargs=job_data['proprio_encoder'],
        policy_config=job_data['policy_config'],
        policy_encoder_kwargs=job_data['policy_encoder'],
        image_encoder_kwargs=job_data['image_encoder_kwargs'],
        language_config=job_data['language_config'],
        resnet_film_config=job_data['resnet_film_config'],
        mdetr_config=job_data.get('mdetr', None),
        train_dl=train_dl,
        val_dl=val_dl,
        dl_info=dl_info,
        epochs=1,
        make_bc_agent=True,
        camera_names=camera_names,
        job_config=job_data,)

    # Save task descriptions and optionally get language embeddings for all envs
    if lazy_env_creation:
        if env_type == 'metaworld':
            train_env_key = 'train_env_names'
            heldout_env_key = 'heldout_env_names'
        elif env_type == 'rlbench':
            train_env_key = 'train_env_info'
            heldout_env_key = 'heldout_env_info'
        elif env_type == 'realworld':
            train_env_key = 'train_env_info'
            heldout_env_key = 'heldout_env_info'

        if env_type == 'pybullet':
            # TODO(saumya): remove hardcoded
            agent.policy.task_descriptions_by_task_name = dict()
            agent.policy.task_descriptions_by_task_name['env_ballbot_pick_red_block'] = ['Pick red block']

            env_descriptions_by_name = dict()
            env_descriptions_by_name['env_ballbot_pick_red_block'] = ['Pick red block']
            try:
                agent.policy.language_emb_model.cache_task_embedding_for_tasks(env_descriptions_by_name)
            except:
                pass
        else:
            agent.policy.save_task_descriptions_for_lazy_envs(
                env_callable,
                env_type,
                train_env_names_dict[train_env_key],
                train_env_names_dict[heldout_env_key],)
        if not job_data['agent_eval']['use']:
            envs_dict = {}
        else:
            envs_dict = train_env_names_dict
    else:
        agent.policy.save_task_descriptions(envs_dict)

    if env_type == 'rlbench':
        job_data['wandb']['group'] = job_data['wandb']['rlbench_group']
        job_data['wandb']['name'] = job_data['wandb']['rlbench_name']

    elif env_type == 'realworld':
        group = job_data['wandb']['realworld_group']
        group_list = group.split('/')
        env_suite = job_data['realworld_envs']['use']
        hparam_desc = (
            f"{job_data['realworld_envs'][env_suite]['normalize_actions']['use']:0d}_"
            f"{job_data['realworld_envs'][env_suite]['normalize_actions']['type']}"
        )
        group = '/'.join(group_list[:-1] + [hparam_desc, group_list[-1]])
        print(f"Using wandb group: {group}")
        job_data['wandb']['group'] = group
        job_data['wandb']['name'] = job_data['wandb']['realworld_name']

    elif env_type == 'metaworld' and 'flava' in job_data['env_kwargs']['embedding_name']:
        job_data['wandb']['group'] = job_data['wandb']['group_flava']
        job_data['wandb']['name'] = job_data['wandb']['name_flava']


    # Save file explicitly
    print('==== Will write config to dict to save for eval ====')
    yaml_path = Path(hydra_dir) / '.hydra' / 'config_gen.yaml'
    with open(yaml_path, 'w') as yaml_f:
        OmegaConf.save(job_data, yaml_f)
    print('==== Did write config to dict to save for eval ====')
    agent.logger.init_wb(job_data)
    wandb.save(str(Path(hydra_dir) / '.hydra' / '*.yaml'), base_path=str(hydra_dir))

    agent.policy.send_to_device()

    # Run training loop

    agent = run_train_eval_loop(job_data, agent, envs_dict, delete_envs=True)
