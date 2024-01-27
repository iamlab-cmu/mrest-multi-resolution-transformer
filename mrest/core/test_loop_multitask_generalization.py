import os, wandb, random
from omegaconf import DictConfig, ListConfig, OmegaConf
import numpy as np
import torch
from pathlib import Path

from train_loop_with_encoder_multi_task import run_train_eval_loop, make_agent, create_dataloaders
from mrest.utils.env_helpers import (create_envs_from_multitask_config, read_config_for_parameterized_envs,
                                       filter_train_val_task_names, RLBenchEnvVariationData,)

if os.environ.get('USE_RLBENCH') == '1':
    from mrest.utils.env_helpers_rlbench import create_rlbench_env_with_name

    
def recursively_set_eval_config_from_args(org_train_cfg, ft_heldout_cfg):
    for k, v in ft_heldout_cfg.items():
        if k == 'eval_num_traj':
            org_train_cfg[k] = ft_heldout_cfg[k]
            continue
        if isinstance(v, ListConfig):
            org_train_cfg[k] = v
        elif isinstance(v, DictConfig):
            if k == 'param_groups' and k not in org_train_cfg:
                print(f'Parma key not in original train_cfg: {k}. Will set from heldout')
                org_train_cfg['param_groups'] = ft_heldout_cfg['param_groups']
                return
            assert org_train_cfg.get(k) is not None, f'Missing key in original train cfg: {k}'
            recursively_set_eval_config_from_args(org_train_cfg[k], ft_heldout_cfg[k])
        else:
            org_train_cfg[k] = v


def bc_eval_with_finetune_multitask(job_data):

    run_path = job_data['checkpoint']['run_path']
    
    download_path='trained_ckpt'
    os.mkdir(download_path)
    
    ckpt_file = wandb.restore(f'{job_data["checkpoint"]["file"]}', run_path=run_path, root=download_path, replace=True)
    checkpoint = torch.load(ckpt_file.name)
    
    config_file = wandb.restore(str('.hydra/config_gen.yaml'), run_path=run_path, root=download_path, replace=True)
    train_cfg = OmegaConf.load(config_file.name)

    # configure GPUs
    os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    physical_gpu_id = 2 #configure_cluster_GPUs(job_data['env_kwargs']['render_gpu_id'])
    physical_gpu_id = np.random.randint(3)
    train_cfg['env_kwargs']['render_gpu_id'] = physical_gpu_id
    train_cfg['seed'] = job_data['seed']

    # ==== Set seed ====
    seed = job_data['seed']
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Make log dir
    if os.path.isdir(train_cfg['job_name']) == False:
        os.mkdir(train_cfg['job_name'])

    hydra_dir = os.getcwd()
    os.chdir(train_cfg['job_name']) # important! we are now in the directory to save data
    if os.path.isdir('iterations') == False:
        os.mkdir('iterations')
    if os.path.isdir('logs') == False:
        os.mkdir('logs')
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    
    # ==== Update config =====
    train_cfg['tag'] = job_data['tag']
    finetune_cfg  = job_data['override_train_cfg']
    recursively_set_eval_config_from_args(train_cfg, finetune_cfg)

    env_type = train_cfg.get('env_type', 'metaworld')
    env_suite_to_use = train_cfg[f'{env_type}_envs']['use']
    train_cfg['envs']['type'] = train_cfg[f'{env_type}_envs'][env_suite_to_use]['type']
    camera_names = train_cfg[f'{env_type}_envs'][env_suite_to_use].get('camera_names', ['left_cap2'])

    # Update wandb
    # Use same group as train project
    if job_data['add_adapters_for_finetuning']:
        # train_cfg['wandb']['group'] = train_cfg['wandb']['group'] + f'/AD_eval'
        # train_cfg['wandb']['group'] = train_cfg['wandb']['group'] + f'/AD_{finetune_cfg["data_env_type"]}'
        train_cfg['wandb']['group'] = train_cfg['wandb']['group'] + f'/AD'
    else:
        train_cfg['wandb']['group'] = train_cfg['wandb']['group'] + f'/FT' 
    
    if job_data['wandb'].get('override_train_group_name', False):
        print("==== Did override group ====")
        suffix_str = 'AD' if job_data['add_adapters_for_finetuning'] else 'FT'
        train_cfg['wandb']['group'] = job_data['wandb']['group'] + f'/{suffix_str}'

    # train_cfg['wandb']['group'] = train_cfg['wandb']['group'] + f'/38_50_buttonpush' 
    # Update project and run names to include checkpoint info for other project.
    train_cfg['wandb']['project'] = job_data['wandb']['project']
    train_cfg['wandb']['name'] = job_data['wandb']['name']
    train_cfg['job_name'] = job_data['wandb']['name']
    
    # Creating dataset
    if train_cfg['dataL_num_workers'] > 0:
        torch.set_num_threads(train_cfg['dataL_num_workers'])

    evaluation_type = job_data['evaluation_type']
    finetune_cfg = job_data[evaluation_type]

    if job_data['finetune']:
        if env_type == 'metaworld':
            heldout_train_dl, heldout_val_dl, dataloader_info = create_dataloaders(
                train_cfg, use_train_data=False, max_train_tasks=2, percent_val_tasks=0.95,
                max_val_tasks=40, env_type=finetune_cfg['data_env_type'], filter_cfg=finetune_cfg['filter_data_env_cfg'])
        elif env_type == 'rlbench':
            pass
    else:
        heldout_train_dl, heldout_val_dl = [], []

    ## Create agent and environments
    if env_type == 'metaworld':
        envs_dict = dict()
        for env_type_key in finetune_cfg['eval_env_types'].keys():
            if finetune_cfg['eval_env_types'][env_type_key].get('eval_all', False):
                finetune_cfg['eval_env_types'][env_type_key] = None
            envs, heldout_envs = create_envs_from_multitask_config(
                train_cfg, env_type=env_type_key, filter_cfg=finetune_cfg['eval_env_types'][env_type_key])
            envs_dict[env_type_key] = envs
            del heldout_envs
            heldout_envs = []

        env_spec = envs[0].spec

    elif env_type == 'rlbench':
        # Save env_info for each (task, variation). The env_info doesn't provide anything new right
        # now, but could be useful for future?
        envs_dict = dict()
        env_spec = None
        for env_type_key, env_config in finetune_cfg['eval_env_types'].items():
            task_name = env_config['task']
            envs_dict[env_type_key] = []
            for variation in env_config['variations']:
                task_info = RLBenchEnvVariationData(task_name, variation, '0', None)
                envs_dict[env_type_key].append(task_info)
                if env_spec is None:
                    rlbench_camera_names = [x[:-4] if x.endswith('_rgb') else x for x in camera_names]
                    env = create_rlbench_env_with_name(train_cfg, task_info, rlbench_camera_names)
                    env_spec = env.spec
                    env.close()
    else:
        raise ValueError(f'Invalid env type: {env_type}')

    # Eval on heldout envs (which are the true eval envs)
    env_suite_to_use = train_cfg[f'{env_type}_envs']['use']
    train_cfg[f'{env_type}_envs'][env_suite_to_use]['eval_on_heldout_envs'] = False
    train_cfg['log_val'] = False
    train_cfg['env_gif_saver']['use'] = True
    train_cfg['env_gif_saver']['save_env_freq'] = 1

    print(OmegaConf.to_yaml(train_cfg))

    env_kwargs = train_cfg['env_kwargs']

    print('Making agent...')

    # breakpoint()
    # train_cfg['bc_kwargs']['loss_scale'] = 1.0

    agent = make_agent(
        env_spec, # all envs should have same spec (observation_dim, action_dim)
        env_kwargs=env_kwargs,
        bc_kwargs=train_cfg['bc_kwargs'],
        proprio_encoder_kwargs=train_cfg['proprio_encoder'],
        policy_config=train_cfg['policy_config'],
        policy_encoder_kwargs=train_cfg['policy_encoder'],
        image_encoder_kwargs=train_cfg['image_encoder_kwargs'],
        language_config=train_cfg['language_config'],
        resnet_film_config=train_cfg.get('resnet_film_config', None),
        mdetr_config=train_cfg.get('mdetr', None),
        train_dl=heldout_train_dl,
        val_dl=heldout_val_dl,
        epochs=1,
        make_bc_agent=True,
        camera_names=camera_names,)
    
    if job_data['add_adapters_for_finetuning']:
        print('====> Adapters added only for finetuning')
        agent.policy.load_non_adapter_pretrained_state_dict(checkpoint)
    elif job_data['load_checkpoint_state_dict']:
        agent.policy.load_state_dict(checkpoint)
    
    if env_type == 'metaworld':
        num_demos_per_task = train_cfg.metaworld_envs[train_cfg.metaworld_envs.use].num_demos_train_per_task
        assert job_data['override_train_cfg']['num_demos_train_per_task'] == num_demos_per_task, (
            f"Number of demos per task in the overriden config are not same as in dataloader. {train_cfg.metaworld_envs.use}, {num_demos_per_task}")
    elif env_type == 'rlbench':
        # How to handle this part?
        num_demos_per_task = 1
        pass
    

    unique_kwargs = {
        'run_id': job_data['run_id'],
        'checkpoint': job_data['checkpoint_name'],
        'finetune': job_data['finetune'],
        'num_demos_per_task': num_demos_per_task,
        'num_tasks': len(train_cfg['envs']['names']),
        'load_checkpoint_state_dict': job_data['load_checkpoint_state_dict'],
        'add_adapters_for_finetuning': job_data['add_adapters_for_finetuning'],
        'seed': job_data['seed'],
        'train_group': train_cfg['wandb']['group'],
        'train_wandb_name': train_cfg['wandb']['name'],
        'data_env_type': finetune_cfg["data_env_type"],
        'env_type': env_type,
        'camera_names': camera_names,
    }
    agent.logger.init_wb(train_cfg, unique_kwargs=unique_kwargs)
    wandb.save(str(Path(hydra_dir) / '.hydra' / '*.yaml'), base_path=str(hydra_dir))
    agent.policy.send_to_device()

    # Save task descriptions and optionally get language embeddings for all envs
    if env_type == 'metaworld':
        lazy_env_creation = False
        envs_data, heldout_envs_data = create_envs_from_multitask_config(
                train_cfg, env_type=finetune_cfg['data_env_type'], filter_cfg=finetune_cfg['filter_data_env_cfg'])
        envs_dict_data = envs_dict.copy()
        envs_dict_data['data'] = envs_data
        agent.policy.save_task_descriptions(envs_dict_data)
        del envs_data
        del heldout_envs_data
        del envs_dict_data
        delete_envs = False

    elif env_type == 'rlbench':
        lazy_env_creation = True
        def rlbench_env_callable(task_info, is_env_type_train: bool):
            rlbench_camera_names = [x[:-4] if x.endswith('_rgb') else x for x in camera_names]
            del is_env_type_train
            return create_rlbench_env_with_name(train_cfg, task_info, rlbench_camera_names)

        env_names_dict = dict()
        env_names = []
        for env_type_key, env_infos in envs_dict.items():
            env_names_dict[env_type_key] = []
            for env_info in env_infos:
                env_names_dict[env_type_key].append(env_info.name)
                env_names.append(env_info.name)
        agent.policy.save_task_descriptions_for_lazy_envs(
            rlbench_env_callable, env_type, 
            [env_info for env_infos in envs_dict.values() for env_info in env_infos], [],)
        envs_dict = env_names_dict
        delete_envs = True

    torch.set_num_threads(1)

    # NOTE: Should we save all gifs?
    # train_cfg['env_gif_saver']['use'] = True
    # train_cfg['env_gif_saver']['save_env_freq'] = 1
    # train_cfg['wandb']['saver_no_eval'] = False
    run_train_eval_loop(train_cfg, agent, envs_dict, train=job_data['finetune'], delete_envs=delete_envs)
    
if __name__ == "__main__":
    bc_test_loop_finetune_encoder_multi_task()
