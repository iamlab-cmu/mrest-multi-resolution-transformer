import os, wandb, random
from omegaconf import OmegaConf, DictConfig
import numpy as np
import time
import torch
from pathlib import Path
from tabulate import tabulate
import pprint

from typing import List, Union

from mrest.core.train_loop_with_encoder_multi_task import (make_agent, eval_on_envs,
                                                             get_reduced_stats_for_evaluation)
from mrest.utils.env_helpers import (create_envs_from_multitask_config, read_config_for_parameterized_envs,
                                       create_single_parameterized_env_with_name, filter_train_val_task_names)
from mrest.utils.encoder_with_proprio import EncoderWithProprio
from mrest.utils.logger import DataLog
from mrest.utils.gspread_utils import SheetsRecorder

from mrest.utils.env_helpers import RLBenchEnvVariationData, RealWorldEnvVariationData
from mrest.utils.env_helpers_realworld import create_realworld_env_with_name

if os.environ.get('USE_RLBENCH') == '1':
    from mrest.utils.env_helpers_rlbench import create_rlbench_env_with_name

if os.environ.get('USE_PYBULLET') == '1':
    from mrest.utils.env_helpers import create_single_pybullet_parameterized_env_with_name


class CheckpointEvaluator:
    def __init__(self, eval_cfg) -> None:
        self.eval_cfg = eval_cfg

        run_path = eval_cfg['checkpoint']['run_path']
        self.run_path = run_path.strip()
        api = wandb.Api(timeout=30)
        self.wandb_run = api.run(self.run_path)

        download_path = f'{run_path}_trained_ckpt_config'
        self.download_path = download_path

        # Load the train config file
        # config_file = wandb.restore(str('.hydra/config_gen.yaml'), run_path=run_path,
        #                             root=download_path, replace=True)
        config_file = wandb.restore(str('.hydra/config_gen.yaml'), run_path=run_path,
                                    root=download_path, replace=True)
        self.train_cfg = OmegaConf.load(config_file.name)
        print('Did load train config')

        # Set the appropriate flags
        self.set_flags()

        # Create the appropriate envs (only if we do not enable lazy env creation)
        if not self.eval_cfg['lazy_env_creation']:
            self.train_envs, self.heldout_envs = self.create_envs()
            self.env_spec = self.train_envs[0].spec

        # init in create_agernt
        self.policy = None
        self.evaluated_checkpoints = []
        self.evaluated_checkpoint_epochs = []

        # Get the row to be used for this result
        self.use_gsheet = False
        if self.use_gsheet:
            gc = SheetsRecorder.create_client()
            self.sheet_recorder = SheetsRecorder(gc, eval_cfg['gsheet']['name'],
                                                eval_cfg['gsheet']['worksheet'])
            self.sheet_info = SheetsRecorder(gc, eval_cfg['gsheet']['name'],
                                            eval_cfg['gsheet']['worksheet_info'],
                                            header_row=1)
            sheet_rows = self.sheet_recorder.find_rows_for_wandb_runpath(run_path)
            if len(sheet_rows):
                if self.eval_cfg['gsheet']['row'] != -1:
                    assert self.eval_cfg['gsheet']['row'] in sheet_rows, (
                        'Google sheet rows does not include provided row.')
                    self.sheet_row = self.eval_cfg['gsheet']['row']
                else:
                    if len(sheet_rows) > 1:
                        print('Found multiple rows with same run path:')
                        print(sheet_rows)
                        raise ValueError('Invalid multiple rows found, provide row to use '
                                        'explicitly through gsheet.row')
                    else:
                        self.sheet_row = sheet_rows[0]
            else:
                raise ValueError(f'Did not find run path in google sheet: {run_path}')

            # NOTE: We want to save info
            # Find run_path and sheet_row in the info worksheet.
            sheet_info_rows = self.sheet_info.find_info_rows_for_wandb_runpath(run_path, self.sheet_row)
            if len(sheet_info_rows):
                # Found sheet row
                assert len(sheet_info_rows) == 1, f'Found more than 1 rows with run path: {run_path}, row: {self.sheet_row}'
                self.sheet_info_row = sheet_info_rows[0]
                print(f'Found existing sheet info for runpath: {run_path}, eval row: {self.sheet_row} '
                    f'row: {self.sheet_info_row}')
            else:
                self.sheet_info_row = self.sheet_info.record_runpath_with_eval_row(run_path, self.sheet_row)
                print(f'Added sheet info for runpath: {run_path}, eval row: {self.sheet_row} '
                    f'row: {self.sheet_info_row}')
        self.did_write_eval_run_info = False



    def get_all_checkpoint_files_from_wandb(self):
        if self.eval_cfg.run_1ckpt.use:
            return [f for f in self.wandb_run.files() if f.name == self.eval_cfg.run_1ckpt.file]
        else:
            extension = ".pth"
            return [f for f in self.wandb_run.files() if f.name.endswith(extension)]

    def update_worksheet_with_eval_value(self, value: float, eval_epoch: int):

        if not self.use_gsheet:
            return

        save_freq_epochs = self.train_cfg['wandb']['saver_no_eval']['save_freq_epochs']
        eval_epoch_idx = eval_epoch // save_freq_epochs
        status = self.sheet_recorder.update_cell_with_value(
            self.sheet_row, eval_epoch_idx, eval_epoch, value
        )
        print(status)
        if not self.did_write_eval_run_info:
            # Update run info
            run_epoch_info = self.eval_cfg['run_epoch']
            current_run_idx = run_epoch_info['current'] if run_epoch_info['use'] else 0
            result_dir = os.getcwd()
            wandb_url = self.current_wandb_run.get_url()
            status = self.sheet_info.record_info_for_eval_run(
                self.sheet_info_row, 5 + 2 * current_run_idx,
                self.eval_cfg['seed'], wandb_url, result_dir)
            print(status)
            self.did_write_eval_run_info = True

    def set_flags(self):
        physical_gpu_id = 2
        physical_gpu_id = self.eval_cfg['gpu_id']
        self.train_cfg['env_kwargs']['render_gpu_id'] = physical_gpu_id

        ## Update config
        self.train_cfg['tag'] = self.eval_cfg['tag']

        # NOTE: Do more evaluations on the single train env.
        if self.eval_cfg.get('eval_num_traj') is None:
            self.train_cfg['eval_num_traj'] = {'train': 3, 'heldout': 1}
        else:
            self.train_cfg['eval_num_traj'] = self.eval_cfg['eval_num_traj']

        self.train_cfg['env_kwargs']['episode_len'] = self.eval_cfg['envs']['episode_len']
        
        # Env gif for visualizing results.
        self.train_cfg['env_gif_saver']['use'] = True
        self.train_cfg['env_gif_saver']['save_env_freq'] = self.eval_cfg['env_gif_saver']['save_env_freq']

    def create_envs(self):
        env_type = self.train_cfg.get('env_type', 'metaworld')
        envs_to_use = self.train_cfg[f'{env_type}_envs']['use']
        self.train_cfg['envs']['type'] = self.train_cfg[f'{env_type}_envs'][envs_to_use]['type']

        # Get the saved train config names
        print('==== Train envs ====')
        print(self.train_cfg['envs']['names'] )
        print('==== Eval envs ====')
        print(self.train_cfg['envs']['heldout_env_names'] )

        ## Create agent and environments
        envs, heldout_envs = create_envs_from_multitask_config(self.train_cfg, env_type='train')

        return envs, heldout_envs

    def create_agent(self):
        train_cfg = self.train_cfg
        env_kwargs = train_cfg['env_kwargs']
        env_type = self.train_cfg.get('env_type', 'metaworld')
        env_suite_to_use = self.train_cfg[f'{env_type}_envs']['use']
        env_suite_config = self.train_cfg[f'{env_type}_envs'][env_suite_to_use]
        camera_names = env_suite_config.get('camera_names', ['left_cap2'])

        if self.eval_cfg['lazy_env_creation']:

            if env_type == 'metaworld':
                env_config_dict_by_type = read_config_for_parameterized_envs(
                    self.train_cfg.data_dir, read_all_configs=True)
                env_config_dict = env_config_dict_by_type['train']

            def metaworld_env_callable(env_name: str, is_env_type_train: bool):
                del is_env_type_train
                return create_single_parameterized_env_with_name(
                    self.train_cfg, env_name, env_config_dict, camera_names=camera_names)
            
            def pybullet_env_callable(env_name: str, is_env_type_train: bool):
                del is_env_type_train
                return create_single_pybullet_parameterized_env_with_name(
                    self.train_cfg, env_name, env_config_dict, camera_names=camera_names)

            def rlbench_env_callable(task_name_with_variation: Union[str, RLBenchEnvVariationData],
                                     is_env_type_train: bool):
                del is_env_type_train
                if isinstance(task_name_with_variation, str):
                    if '_data' in task_name_with_variation:
                        task_name_with_variation = task_name_with_variation.split('_data_')[0]
                    task_name, variation = task_name_with_variation.split('_var_')
                elif isinstance(task_name_with_variation, RLBenchEnvVariationData):
                    task_name = task_name_with_variation.env
                    variation = task_name_with_variation.variation
                else:
                    raise ValueError("Cannot create RLBench task name and variation")
                camera_names = ['front', 'wrist']
                task_info = RLBenchEnvVariationData(task_name, int(variation), '0', None)
                return create_rlbench_env_with_name(self.train_cfg, task_info, camera_names)

            def realworld_env_callable(task_name_with_variation: Union[str, RLBenchEnvVariationData],
                                       is_env_type_train: bool):
                del is_env_type_train
                if isinstance(task_name_with_variation, str):
                    if '_data' in task_name_with_variation:
                        task_name_with_variation = task_name_with_variation.split('_data_')[0]
                    if '_var_' in task_name_with_variation:
                        task_name, variation = task_name_with_variation.split('_var_')
                    else:
                        task_name = task_name_with_variation
                        variation = '0'

                elif isinstance(task_name_with_variation, RealWorldEnvVariationData):
                    task_name = task_name_with_variation.env
                    variation = task_name_with_variation.variation
                else:
                    raise ValueError("Cannot create RLBench task name and variation")
                camera_names = ['hand', 'static']
                task_info = RealWorldEnvVariationData(task_name, int(variation), '0', None)
                return create_realworld_env_with_name(self.train_cfg, task_info, camera_names)

            def env_callable(*args, **kwargs):
                if env_type == 'metaworld':
                    return metaworld_env_callable(*args, **kwargs)
                elif env_type == 'rlbench':
                    return rlbench_env_callable(*args, **kwargs)
                elif env_type == 'realworld':
                    return realworld_env_callable(*args, **kwargs)
                elif env_type == 'pybullet':
                    return pybullet_env_callable(*args, **kwargs)
                else:
                    return None

            if env_type == 'pybullet':
                # TODO: remove hardcoded
                self.env_spec = OmegaConf.create({"observation_dim":4, "action_dim": 4, "horizon": 200})
            else:
                env = env_callable(self.train_cfg['envs']['names'][0], True)
                self.env_spec = env.spec
                if env_type == 'rlbench':
                    env.close()
                del env

        policy = make_agent(
            self.env_spec , # all envs should have same spec (observation_dim, action_dim)
            env_kwargs=env_kwargs,
            bc_kwargs=train_cfg['bc_kwargs'],
            proprio_encoder_kwargs=train_cfg['proprio_encoder'],
            policy_config=train_cfg['policy_config'],
            policy_encoder_kwargs=train_cfg['policy_encoder'],
            image_encoder_kwargs=train_cfg['image_encoder_kwargs'],
            language_config=train_cfg['language_config'],
            mdetr_config=train_cfg.get('mdetr', None),
            resnet_film_config=train_cfg.get('resnet_film_config', None),
            make_bc_agent=False,
            camera_names=camera_names,
            job_config=train_cfg,)

        # Save task descriptions and optionally get language embeddings for all envs
        
        policy.send_to_device()
        if self.eval_cfg['lazy_env_creation']:

            try:
                train_envs = self.train_cfg['envs']['names']
                heldout_envs = self.train_cfg['envs']['heldout_env_names']

            except:
                if env_type == 'metaworld' or env_type == 'pybullet':
                    env_config_dict_by_type = read_config_for_parameterized_envs(
                        self.train_cfg.data_dir, read_all_configs=True)
                    env_config_dict = env_config_dict_by_type['train']
                    train_task_names = filter_train_val_task_names(env_config_dict)
                    train_envs, heldout_envs = train_task_names, train_task_names
                    self.train_cfg['envs']['names'] = train_envs
                    self.train_cfg['envs']['heldout_env_names'] = heldout_envs
                elif env_type == 'rlbench':
                    raise NotImplementedError

            if set(train_envs) == set(heldout_envs):
                heldout_envs = []
            if env_type == 'metaworld':
                policy.save_task_descriptions_for_lazy_envs(env_callable, env_type, train_envs, heldout_envs)
            elif env_type == 'pybullet':
                # TODO: remove hardcoded
                policy.task_descriptions_by_task_name = dict()
                policy.task_descriptions_by_task_name['env_ballbot_pick_red_block'] = ['Pick red block']

                env_descriptions_by_name = dict()
                env_descriptions_by_name['env_ballbot_pick_red_block'] = ['Pick red block']
                try:
                    policy.language_emb_model.cache_task_embedding_for_tasks(env_descriptions_by_name)
                except:
                    pass
            elif env_type == 'rlbench':
                train_env_info, heldout_env_info = [], []
                for n in train_envs:
                    if '_data_' in n:
                        n = n.split('_data_')[0]
                    train_env_info.append(RLBenchEnvVariationData(n.split('_var_')[0], int(n.split('_var_')[1]), '0', None))
                for n in heldout_envs:
                    if '_data_' in n:
                        n = n.split('_data_')[0]
                    heldout_env_info.append(RLBenchEnvVariationData(n.split('_var_')[0], int(n.split('_var_')[1]), '0', None))
                policy.save_task_descriptions_for_lazy_envs(env_callable, env_type, train_env_info, heldout_env_info)
            elif env_type == 'realworld':
                train_env_info, heldout_env_info = [], []
                for n in train_envs:
                    if '_data_' in n:
                        n = n.split('_data_')[0]
                    if '_var_' in n:
                        train_env_info.append(RealWorldEnvVariationData(n.split('_var_')[0], int(n.split('_var_')[1]), '0', None))
                    else:
                        train_env_info.append(RealWorldEnvVariationData(n, 0, '0', None))
                policy.save_task_descriptions_for_lazy_envs(env_callable, env_type, train_env_info, heldout_env_info)

        else:
            envs_dict = {'train': self.train_envs, 'heldout': self.heldout_envs}
            policy.save_task_descriptions(envs_dict)

            del self.train_envs
            del self.heldout_envs

        # Do not save the policy on cpu
        policy.send_to_device('cpu')
        torch.cuda.empty_cache()

        self.policy = policy

    def get_wandb_group_suffix(self):
        if self.eval_cfg['run_epoch']['use']:
            current_run = self.eval_cfg['run_epoch']['current']
            total_runs = self.eval_cfg['run_epoch']['total']
            return f'{current_run}_{total_runs}' if current_run > 0 else f'{total_runs}_{total_runs}'
        else:
            return 'all'

    def start(self):
        print('Will start eval loop')

        # Create logger
        logger = DataLog()

        wandb_components = self.run_path.split('/')
        if len(wandb_components) == 3:
            entity, project, run_id = wandb_components
        elif len(wandb_components) == 4:
            assert wandb_components[2] == 'runs'
            entity, project, _, run_id = wandb_components
        else:
            raise ValueError(f'Invalid run path: {self.run_path}')

        assert self.train_cfg['wandb']['project'] == project, 'Incorrect project'
        assert self.train_cfg['wandb']['entity'] == entity, 'Incorrect entity'
        train_group = self.train_cfg['wandb']['group']
        if train_group.startswith('saumya') or train_group.startswith('train'):
            new_group = 'eval/' + '/'.join(train_group.split('/')[1:])
        else:
            new_group = train_group
        # NOTE: Update project to eval for these runs (since we have too many of them).
        self.train_cfg['wandb']['project'] = self.eval_cfg['wandb']['project']
        self.train_cfg['wandb']['group'] = new_group
        self.train_cfg['wandb']['name'] += f'/run_{self.get_wandb_group_suffix()}'
        self.logger = logger

        # Check if we only have to run some checkpoints
        run_all = True
        if self.eval_cfg['run_epoch']['use']:
            run_all = False
            current_run = self.eval_cfg['run_epoch']['current']
            total_runs = self.eval_cfg['run_epoch']['total']
            assert current_run >= 0 and current_run < total_runs
            train_ckpt_freq = self.train_cfg['wandb']['saver_no_eval']['save_freq_epochs']

        self.current_wandb_run = logger.init_wb(self.train_cfg)

        time_since_last_checkpoint_evaluated = time.time()

        # Sort new checkpoints based on epochs and evaluate them based on that.
        def _get_checkpoint_epoch_from_name(x: str) -> int:
            try:
                ckpt = int(x.split('/')[1].split('_')[1].split('.')[0])
            except:
                ckpt = int(x.split('/')[1].split('_')[3][5:])
            return ckpt

        # NOTE: Do not evaluate checkpoints beyond this epoch. While we can add
        # checkpoints that are logged (by looking at log_freq), here we just add
        # every possible checkpoint.
        start_epoch_from = self.eval_cfg['run_epoch'].get('start_eval_from', 0)
        for i in range(start_epoch_from):
            self.evaluated_checkpoints.append(f'checkpoints/ckpt_{i:04d}.pth')
            self.evaluated_checkpoint_epochs.append(i)

        while True:

            # Check if we have really evaluated all checkpoints for this run (and if yes exit).
            if len(self.evaluated_checkpoint_epochs) > 0:
                last_epoch_evaluated = self.evaluated_checkpoint_epochs[-1]
                next_possible_run_to_eval = (1 if not self.eval_cfg['run_epoch']['use']
                                             else self.eval_cfg['run_epoch']['current'])
                if self.train_cfg['epochs'] <= last_epoch_evaluated + next_possible_run_to_eval:
                    print(f'Total train epochs: {self.train_cfg["epochs"]}, '
                          f'last epoch evaluated: {last_epoch_evaluated}, '
                          f'next epoch does not exist: {last_epoch_evaluated + next_possible_run_to_eval}. Hence will exit!!')
                    break

            all_checkpoints = self.get_all_checkpoint_files_from_wandb()
            new_checkpoints_to_eval = []
            for ckpt in all_checkpoints:
                # Check if this run will indeed evaluate this checkpoint, if not skip adding it to
                # new checkpionts to evaluate.
                if not run_all:
                    checkpoint_epoch = _get_checkpoint_epoch_from_name(ckpt.name)
                    if (checkpoint_epoch // train_ckpt_freq) % total_runs != current_run:
                        continue
                if ckpt.name not in self.evaluated_checkpoints:
                    new_checkpoints_to_eval.append(ckpt)

            # No new checkpoint to evaluate sleep for now
            if len(new_checkpoints_to_eval) == 0:
                wait_time = time.time() - time_since_last_checkpoint_evaluated
                wait_before_terminate_time_in_mins = self.eval_cfg.get('wait_before_terminate_time_in_mins', 300)
                if wait_time >= (wait_before_terminate_time_in_mins * 60):
                    print(f'Last checkpoint eval at: {time_since_last_checkpoint_evaluated}, '
                          f'current time: {time.time()}, no checkpoint wait: {wait_time // 60:.2f} mins')
                    break
                time.sleep(self.eval_cfg['sleep_time'])
                continue

            checkpoints_with_epochs = []
            for checkpoint in new_checkpoints_to_eval:
                checkpoint_epoch = _get_checkpoint_epoch_from_name(checkpoint.name)
                if run_all:
                    checkpoints_with_epochs.append((checkpoint_epoch, checkpoint))
                else:
                    assert checkpoint_epoch % train_ckpt_freq == 0, 'Invalid train checkpoint'
                    if (checkpoint_epoch // train_ckpt_freq) % total_runs == current_run:
                        checkpoints_with_epochs.append((checkpoint_epoch, checkpoint))
            checkpoints_with_epochs = sorted(checkpoints_with_epochs, key=lambda x: x[0])

            print(f'Found {len(checkpoints_with_epochs)} to run.')

            for ckpt_epoch, ckpt in checkpoints_with_epochs:
                eval_start_time = time.time()
                print('Will run eval for checkpoint:')
                print(f'\t \t  ckpt: {ckpt.name}')
                print(f'\t \t epoch: {ckpt_epoch}')
                download_path = f'{self.run_path}_ckpt_{ckpt_epoch}.pth'
                print("Downloading...")
                ckpt_file = wandb.restore(ckpt.name, run_path=self.run_path,
                                          root=download_path, replace=False)
                checkpoint = torch.load(ckpt_file.name)

                print("Did download will now run")
                self.run_eval_with_checkpoint_file(checkpoint, ckpt_epoch)
                eval_end_time = time.time()
                print(f'Did run eval for checkpoint, time elapsed: {eval_end_time - eval_start_time:.2f}')

                # Add to evaluated checkpoints.
                self.evaluated_checkpoints.append(ckpt.name)
                self.evaluated_checkpoint_epochs.append(ckpt_epoch)
                print(f'Total evaluations run: {len(self.evaluated_checkpoints)}')

                del checkpoint
                torch.cuda.empty_cache()

                # Update time
                time_since_last_checkpoint_evaluated = time.time()


    def run_eval_with_checkpoint_file(self, checkpoint, ckpt_epoch: int):
        # ==== Set seed ====
        seed = self.eval_cfg['seed']
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # NOTE: Do more evaluations on the single train env.
        # train_cfg['eval_num_traj'] = ft_heldout_cfg['eval_num_traj']
        # self.train_cfg['eval_num_traj'] = {'train': 1, 'heldout': 0}

        # Get checkpoint epoch from checkpoint file
        self.policy.load_state_dict(checkpoint)
        self.policy.send_to_device('cuda')
        self.run_eval(ckpt_epoch)
        self.policy.send_to_device('cpu')
        torch.cuda.empty_cache()

    def filter_single_data_envs_for_eval(self, env_list: List[str]):
        data_independent_env_keys = []
        data_independent_envs = []
        for env_name in env_list:
            if '_var_' in env_name and '_data_' in env_name:
                data_independent_env = env_name.split('_data_')[0]
                if data_independent_env not in data_independent_env_keys:
                    data_independent_env_keys.append(data_independent_env)
                    data_independent_envs.append(env_name)

        return data_independent_envs, data_independent_env_keys

    def run_eval(self, checkpoint_epoch: int):
        job_data = self.train_cfg
        env_suite_type = job_data.get('env_type', 'metaworld')
        env_suite_to_use = job_data[f'{env_suite_type}_envs']['use']
        record_per_env_eval = job_data[f'{env_suite_type}_envs'][env_suite_to_use]['record_per_env_eval']

        eval_on_train_envs = job_data[f'{env_suite_type}_envs'][env_suite_to_use].get('eval_on_train_envs', True)
        # eval_on_heldout_envs = job_data['metaworld_envs'][env_suite_to_use]['eval_on_heldout_envs']
        eval_on_heldout_envs = False

        if self.eval_cfg['lazy_env_creation']:
            if env_suite_type == 'metaworld' or env_suite_type == 'pybullet':
                train_envs = self.train_cfg['envs']['names']
                heldout_envs = self.train_cfg['envs']['heldout_env_names']
            elif env_suite_type == 'rlbench':
                train_envs, _ = self.filter_single_data_envs_for_eval(self.train_cfg['envs']['names'])
                heldout_envs, _ = self.filter_single_data_envs_for_eval(self.train_cfg['envs']['heldout_env_names'])
            elif env_suite_type == 'realworld':
                train_envs = self.train_cfg['envs']['names']
                heldout_envs = self.train_cfg['envs']['heldout_env_names']
            else:
                raise ValueError(f'Invalid env suite: {env_suite_type}')
        else:
            train_envs, heldout_envs = self.create_envs()

        np.random.seed(self.eval_cfg['seed'] + checkpoint_epoch)

        if env_suite_type == 'metaworld' or env_suite_type == 'pybullet':
            env_config_dict_by_type = read_config_for_parameterized_envs(
                self.train_cfg.data_dir, read_all_configs=True)
            env_config_dict = env_config_dict_by_type['train']
        elif env_suite_type == 'rlbench':
            # We don't have any separate config for RLBench environments
            env_config_dict = None
            pass
        elif env_suite_type == 'realworld':
            env_config_dict = None

        print("Train envs:")
        pprint.pprint(train_envs)
        print("Eval envs:")
        pprint.pprint(heldout_envs)

        self.policy.eval()
        # Change accordingly.
        # train_envs ['lift_blue_block_var_0_data_0', 'lift_green_block_var_0_data_0', 'lift_yellow_block_var_0_data_0']
        for envs, should_eval_on_envs, env_type in zip([train_envs, heldout_envs],
                                                       [eval_on_train_envs, eval_on_heldout_envs],
                                                       ['train', 'heldout']):
            if not should_eval_on_envs:
                continue

            if (isinstance(self.eval_cfg['eval_num_traj'], dict) or
                isinstance(self.eval_cfg['eval_num_traj'], DictConfig)):
                num_episodes = self.eval_cfg['eval_num_traj'][env_type]
            else:
                num_episodes = self.eval_cfg['eval_num_traj']

            (success_percent, any_success_percent, mean_reward) = eval_on_envs(
                envs, self.policy, self.logger, job_data, record_per_env_eval, num_episodes,
                checkpoint_epoch, checkpoint_epoch, delete_envs=True,
                env_config_dict=env_config_dict,
                set_numpy_seed_per_env=False,)

            self.logger.log_kv(f'eval_success/all_{env_type}_envs', np.mean(success_percent))
            self.logger.log_kv(f'eval_any_success/all_{env_type}_envs', np.mean(any_success_percent))
            self.logger.log_kv(f'eval_reward/all_{env_type}_envs', np.mean(mean_reward))

            # Write to google sheet
            if env_type == 'train':
                self.update_worksheet_with_eval_value(np.mean(any_success_percent), checkpoint_epoch)

        self.logger.log_kv('eval_epoch', checkpoint_epoch)

        # Get the set of statistics we want and log them.
        # NOTE: While we only care about final statistics we save them repeatedly so that in
        # case of any crash we can still get some relevant values.
        final_success_log = get_reduced_stats_for_evaluation(self.logger)
        for k, v in final_success_log.items():
            self.logger.log_kv(k, v)
        self.logger.save_wb(step=checkpoint_epoch, filter_step_key=True)

        print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                    self.logger.get_current_log().items()))
        print(tabulate(print_data))


def bc_eval_loop_on_train_checkpoints(job_data):
    run_path = job_data['checkpoint']['run_path']
    print(f'Will evaluate checkpoint: {run_path}')

    torch.set_num_threads(1)

    ckpt_evaluator = CheckpointEvaluator(job_data)
    ckpt_evaluator.create_agent()
    ckpt_evaluator.start()


if __name__ == "__main__":
    bc_eval_loop_on_train_checkpoints()
