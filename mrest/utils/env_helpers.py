from collections import namedtuple
from pathlib import Path
from typing import Any, List, Mapping, Optional
from dataclasses import dataclass

from omegaconf import DictConfig, ListConfig

import os
import pickle

from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

from mrest.utils.gym_env import GymEnv
from mrest.utils.obs_wrappers import MuJoCoDictObs
from mrest.gym_utils.multistep_wrapper import MultiStepWrapper

if os.environ.get('USE_RLBENCH') == '1':
    from mrest.utils.env_helpers_rlbench import create_rlbench_env_with_name

if os.environ.get('USE_PYBULLET') == '1':
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

# if os.environ.get('USE_MUJOCO_23') == '1':
from metaworld.envs.mujoco2_3.sawyer_xyz.sawyer_pick_place_multitask_procedural import SawyerPickAndPlaceMultiTaskEnvProcV2

if os.environ.get('USE_PYBULLET') == '1':
    from robot.robot_simulator_pickup_env import RobotSimulatorPickup
    from mrest.utils.obs_wrappers_pybullet import BallbotDictObs


@dataclass
class RealWorldEnvVariationData:
    env: str
    variation: int
    data_key: str
    data_dir: str

    @property
    def name(self) -> str:
        return f'{self.env}_var_{self.variation}_data_{self.data_key}'
    
    @property
    def task_variation_name(self) -> str:
        return f'{self.env}_var_{self.variation}'


@dataclass
class RLBenchEnvVariationData:
    env: str
    variation: int
    data_key: str
    data_dir: str

    @property
    def name(self) -> str:
        return f'{self.env}_var_{self.variation}_data_{self.data_key}'
    
    @property
    def task_variation_name(self) -> str:
        return f'{self.env}_var_{self.variation}'
    

def get_metaworld_target_object_name(env_config: Mapping[str, Any]):
    if 'block' in env_config['target_object']:
        target_obj = env_config['blockA_config']['color']
    elif 'stick' in env_config['target_object']:
        target_obj = env_config['stickA_config']['color']
    elif 'faucetBase' in env_config['target_object']:
        if env_config.get('target_object_color_cfg',None) is not None:
            faucetHandleColor = env_config['target_object_color_cfg'].get('faucetHandleColor', 'red')
            faucetHeadColor = env_config['target_object_color_cfg'].get('faucetHead1', 'grey')
        else:
            faucetHandleColor = 'red'
            faucetHeadColor = 'grey'
        target_obj = f'faucetBase_handlecolor_{faucetHandleColor}_headcolor_{faucetHeadColor}'
    elif 'peg' in env_config['target_object']:
        if env_config.get('target_object_color_cfg',None) is not None:
            PegCylinderColor = env_config['target_object_color_cfg'].get('PegCylinder', 'red')
            WrenchHandleColor = env_config['target_object_color_cfg'].get('WrenchHandle', 'green')
        else:
            PegCylinderColor = 'red'
            WrenchHandleColor = 'green'
        target_obj = f'peg_pegcolor_{PegCylinderColor}_wrenchcolor_{WrenchHandleColor}'
    elif 'RoundNut' in env_config['target_object']:
        if env_config.get('target_object_color_cfg',None) is not None:
            PegCylinderColor = env_config['target_object_color_cfg'].get('PegCylinder', 'red')
            WrenchHandleColor = env_config['target_object_color_cfg'].get('WrenchHandle', 'green')
        else:
            PegCylinderColor = 'red'
            WrenchHandleColor = 'green'
        target_obj = f'RoundNut_pegcolor_{PegCylinderColor}_wrenchcolor_{WrenchHandleColor}'
    else:
        target_obj = env_config['target_object']
    
    return target_obj


def filter_train_val_task_names(env_config_dict, filter_cfg=None):
    """Filter parameterized tasks into train and validation tasks."""
    use_filter = filter_cfg is not None

    # Optionally override using 'use' flag within the filter.
    if use_filter:
        use_filter = filter_cfg.get('use', True)
    
    if use_filter:
        print("Will filter out some envs.")

        # initialize all skill-color pairs to zero
        target_objs = dict()
        for target_obj in filter_cfg['target_objs']:
            target_objs.update({target_obj: 0})
        st_pairs = dict()
        for skill in filter_cfg['skills']:
            st_pairs.update({skill:target_objs.copy()})
        
        target_obj_skill_pairs_dict = dict()

        task_names = []
        max_envs = len(filter_cfg['skills'])*len(filter_cfg['target_objs'])*filter_cfg['max_envs_per_skill_target_obj_pair']
        for env_name, env_config in env_config_dict.items():
            target_obj = get_metaworld_target_object_name(env_config)
            skill = env_config['skill']
            if target_obj in filter_cfg['target_objs'] and skill in filter_cfg['skills']:

                if st_pairs[skill][target_obj] == filter_cfg['max_envs_per_skill_target_obj_pair']:
                    continue
                else:
                    st_pairs[skill][target_obj] += 1
                    task_names.append(env_name)
                
                target_obj_skill_tuple = (target_obj, skill)
                if target_obj_skill_pairs_dict.get(target_obj_skill_tuple) is None:
                    target_obj_skill_pairs_dict[target_obj_skill_tuple] = 0
                target_obj_skill_pairs_dict[target_obj_skill_tuple] += 1

            else:
                print(f"Skipping: target_obj {target_obj}, skill: {skill}")

        return task_names
    else:
        return [task_name for task_name in env_config_dict.keys()]

        
def read_config_for_parameterized_envs(config_path: str, read_all_configs: bool = False,
                                       read_pickle: bool = True):
    env_config = Path(config_path)
    assert env_config.exists(), f'Env config does not exist: {env_config}'

    env_config_path_suffix = 'env_configs.pkl' if read_pickle else 'env_configs.yaml'
    configs = [p for p in env_config.iterdir() if p.name.endswith(env_config_path_suffix)]

    if read_all_configs:
        configs_by_type = {}
        for config in configs:
            with open(str(config), 'rb') as env_f:
                config_dict = pickle.load(env_f)
                config_name = config.name.split('_env_configs')[0]
                configs_by_type[config_name] = config_dict

        return configs_by_type
    else:
        train_env_config = env_config / 'train_env_configs.pkl'
        eval_env_config = env_config / 'eval_env_configs.pkl'
        with open(str(train_env_config), 'rb') as env_f:
            train_env_config_dict = pickle.load(env_f)
        with open(str(eval_env_config), 'rb') as env_f:
            eval_env_config_dict = pickle.load(env_f)

        return train_env_config_dict, eval_env_config_dict


def create_single_parameterized_env_with_name(job_data: Mapping[str, Any], env_name: str, env_config_dict: Mapping[str, Any], camera_names):
    env_suite_to_use = job_data['metaworld_envs']['use']
    is_parameterized_env = job_data['metaworld_envs'][env_suite_to_use]['is_parameterized']

    env_kwargs = job_data['env_kwargs']
    common_env_kwargs = {
        'image_width': env_kwargs['image_width'],
        'image_height': env_kwargs['image_height'],
        'pixel_based': env_kwargs['pixel_based'],
        'render_gpu_id': env_kwargs['render_gpu_id'],
        'proprio': env_kwargs['proprio'],
        'lang_cond': env_kwargs['lang_cond'],
        'gc': env_kwargs['gc'],
        'episode_len': env_kwargs['episode_len'],
        'multi_task': job_data['multi_task'],
        # Sensors that define at what frequency should we be using them.
        'multi_temporal_sensors': job_data.get('multi_temporal_sensors', None),
    }

    assert is_parameterized_env

    if 'mt_sawyer' in env_suite_to_use and 'multiobj_multiskill' in env_suite_to_use:
        env = sawyer_multiobj_multitask_env_constructor(
            env_name=env_name,
            env_type=env_suite_to_use,
            env_config=env_config_dict[env_name],
            camera_names=camera_names,
            **common_env_kwargs)
    
    elif 'mt_sawyer' in env_suite_to_use and 'multiview_robustness' in env_suite_to_use:
        env = sawyer_pick_and_place_multitask_multiview_robustness_env_constructor(
            env_name=env_name,
            env_type=env_suite_to_use,
            env_config=env_config_dict[env_name],
            camera_names=camera_names,
            **common_env_kwargs)

    elif ('mt_sawyer' in env_suite_to_use and 
         ('multiview_robustness' in env_suite_to_use or 'multiview_procedural' in env_suite_to_use)):
        # For procedural objects
        env = sawyer_pick_and_place_multitask_multiview_robustness_env_constructor(
            env_name=env_name,
            env_type=env_suite_to_use,
            env_config=env_config_dict[env_name],
            camera_names=camera_names,
            **common_env_kwargs)

    elif 'mt_sawyer' in env_suite_to_use and 'multiskill' in env_suite_to_use:
        env = sawyer_pick_and_place_multitask_multiskill_env_constructor(
            env_name=env_name,
            env_type=env_suite_to_use,
            env_config=env_config_dict[env_name],
            camera_names=camera_names,
            **common_env_kwargs)

    elif 'mt_sawyer' in env_suite_to_use:
        env = sawyer_pick_and_place_multitask_env_constructor(
            env_name=env_name,
            env_type=env_suite_to_use,
            env_config=env_config_dict[env_name],
            camera_names=camera_names,
            **common_env_kwargs)

    else:
        raise ValueError('Invalid env suite')

    return env

def create_rlbench_envs_from_multitask_config(job_data, env_type: str = 'train',
                                              filter_cfg: Optional[DictConfig] =None,
                                              return_only_env_names: bool = False):

    env_suite_to_use = job_data['rlbench_envs']['use']
    env_suite_config = job_data['rlbench_envs'][env_suite_to_use]

    env_kwargs = {}

    # data_dir should be populated correctly before this method is being called.
    train_env_info = []
    for task_name, task_info in env_suite_config.data_dirs.items():
        eval_variations = []
        for data_key, data_config in task_info.items():
            for eval_variation in data_config['eval_variations']:
                if eval_variation not in eval_variations:
                    eval_variations.append(eval_variation)
        for eval_variation in eval_variations:
            # Data key (data_0) is not important since we are only going to evaluate using thse envs.
            train_env_info.append(RLBenchEnvVariationData(task_name, eval_variation, '0', None))

    # train_env_names = filter_train_val_task_names(env_config_dict, filter_cfg)

    print(f'==== Eval envs of type: {env_type} ====')
    print([env_info.name for env_info in train_env_info])

    return {
        'train_env_info': train_env_info,
        'heldout_env_info': [],
    }


def create_realworld_envs_from_multitask_config(job_data, env_type: str = 'train',
                                                filter_cfg: Optional[DictConfig] =None,
                                                return_only_env_names: bool = False):

    env_suite_to_use = job_data['realworld_envs']['use']
    env_suite_config = job_data['realworld_envs'][env_suite_to_use]

    env_kwargs = {}

    # data_dir should be populated correctly before this method is being called.
    train_env_info = []
    for task_name, task_info in env_suite_config.data_dirs.items():
        eval_variations = [0]
        for eval_variation in eval_variations:
            # Data key (data_0) is not important since we are only going to evaluate using thse envs.
            train_env_info.append(RealWorldEnvVariationData(task_name, eval_variation, '0', None))

    # train_env_names = filter_train_val_task_names(env_config_dict, filter_cfg)

    print(f'==== Eval envs of type: {env_type} ====')
    print([env_info.name for env_info in train_env_info])

    return {
        'train_env_info': train_env_info,
        'heldout_env_info': [],
    }


def create_pybullet_envs_from_multitask_config(job_data: Mapping[str, Any], env_type: str = 'train',
                                      filter_cfg=None,
                                      return_only_env_names: bool = False):
    '''Create envs from config.

    Args:

    job_data: Config for the given job.
    use_train_param_env: Should we use train config or eval config. This flag is only used
    return_only_env_names: Set to true if we only want to get env names and lazily create envs.
    '''


    env_suite_to_use = job_data['pybullet_envs']['use']
    env_suite_config = job_data['pybullet_envs'][env_suite_to_use]

    env_kwargs = job_data['env_kwargs']
    
    # data_dir should be populated correctly before this method is being called.
    env_config_dict_by_type = read_config_for_parameterized_envs(
        job_data.data_dir, read_all_configs=True)
    env_config_dict = env_config_dict_by_type[env_type]

    train_env_names = filter_train_val_task_names(env_config_dict, filter_cfg)

    print(f'==== Eval envs of type: {env_type} ====')
    print(train_env_names)

    if return_only_env_names:
        return {
            'train_env_names': train_env_names,
            'heldout_env_names': [],
            'env_config_dict': env_config_dict,
        }

    train_envs, heldout_envs = [], []
    camera_names = env_suite_config.get('camera_names', ['left_cap2'])
    for env_name in train_env_names:
        env = create_single_pybullet_parameterized_env_with_name(job_data, env_name, env_config_dict, camera_names)
        train_envs.append(env)

    return train_envs, heldout_envs

def create_single_pybullet_parameterized_env_with_name(job_data: Mapping[str, Any], env_name: str, env_config_dict: Mapping[str, Any], camera_names):
    env_suite_to_use = job_data['pybullet_envs']['use']

    env_kwargs = job_data['env_kwargs']
    common_env_kwargs = {
        'image_width': env_kwargs['image_width'],
        'image_height': env_kwargs['image_height'],
        'pixel_based': env_kwargs['pixel_based'],
        'render_gpu_id': env_kwargs['render_gpu_id'],
        'proprio': env_kwargs['proprio'],
        'lang_cond': env_kwargs['lang_cond'],
        'gc': env_kwargs['gc'],
        'episode_len': env_kwargs['episode_len'],
        'multi_task': job_data['multi_task'],
        # Sensors that define at what frequency should we be using them.
        'multi_temporal_sensors': job_data.get('multi_temporal_sensors', None),
        'min_action': job_data['pybullet_envs'][env_suite_to_use].get('min_action', None),
        'max_action':job_data['pybullet_envs'][env_suite_to_use].get('max_action', None),
        'action_type':job_data['pybullet_envs'][env_suite_to_use].get('action_type', 'delta_obs_pos'),
        'all_low_res': env_kwargs.get('all_low_res', False),
        'all_high_res': env_kwargs.get('all_high_res', False),
    }
    env = ballbot_pickup_multitask_multiview_env_constructor(
                env_name=env_name,
                env_type=env_suite_to_use,
                env_config=env_config_dict[env_name],
                camera_names=camera_names,
                **common_env_kwargs)
    return env
    

def create_envs_from_multitask_config(job_data: Mapping[str, Any], env_type: str = 'train',
                                      filter_cfg=None,
                                      return_only_env_names: bool = False):
    '''Create envs from config.

    Args:

    job_data: Config for the given job.
    use_train_param_env: Should we use train config or eval config. This flag is only used
    return_only_env_names: Set to true if we only want to get env names and lazily create envs.
    '''

    if job_data.get('env_type', 'metaworld') == 'rlbench':
        return create_rlbench_envs_from_multitask_config(
            job_data, env_type=env_type, filter_cfg=filter_cfg, return_only_env_names=return_only_env_names)
    elif job_data.get('env_type', 'metaworld') == 'realworld':
        return create_realworld_envs_from_multitask_config(
            job_data, env_type=env_type, filter_cfg=filter_cfg, return_only_env_names=return_only_env_names)

    if job_data.get('env_type', 'metaworld') == 'rlbench':
        return create_pybullet_envs_from_multitask_config(
            job_data, env_type=env_type, filter_cfg=filter_cfg, return_only_env_names=return_only_env_names)

    env_suite_to_use = job_data['metaworld_envs']['use']
    env_suite_config = job_data['metaworld_envs'][env_suite_to_use]
    is_parameterized_env = job_data['metaworld_envs'][env_suite_to_use]['is_parameterized']

    env_kwargs = job_data['env_kwargs']
    common_env_kwargs = {
        'image_width': env_kwargs['image_width'],
        'image_height': env_kwargs['image_height'],
        'camera_name': env_kwargs['camera_name'],
        'pixel_based': env_kwargs['pixel_based'],
        'render_gpu_id': env_kwargs['render_gpu_id'],
        'proprio': env_kwargs['proprio'],
        'lang_cond': env_kwargs['lang_cond'],
        'gc': env_kwargs['gc'],
        'episode_len': env_kwargs['episode_len'],
        'multi_task': job_data['multi_task'],
    }

    if is_parameterized_env:
        # data_dir should be populated correctly before this method is being called.
        env_config_dict_by_type = read_config_for_parameterized_envs(
            job_data.data_dir, read_all_configs=True)
        env_config_dict = env_config_dict_by_type[env_type]

        # train_env_names = job_data['envs']['names']
        # heldout_env_names = job_data['envs']['heldout_env_names']

        train_env_names = filter_train_val_task_names(env_config_dict, filter_cfg)

        print(f'==== Eval envs of type: {env_type} ====')
        print(train_env_names)

        if return_only_env_names:
            return {
                'train_env_names': train_env_names,
                'heldout_env_names': [],
                'env_config_dict': env_config_dict,
            }

        train_envs, heldout_envs = [], []
        for env_name in train_env_names:
            env = create_single_parameterized_env_with_name(job_data, env_name, env_config_dict, env_suite_config.get('camera_names', ['left_cap2']))
            train_envs.append(env)

        # for env_name in heldout_env_names:
        #     env = create_single_parameterized_env_with_name(job_data, env_name, env_config_dict)
        #     heldout_envs.append(env)

        return train_envs, heldout_envs

    else:
        train_env_names = job_data['envs']['names']
        heldout_env_names = job_data['envs']['heldout_env_names']

        # TODO: Task descriptions sould come from a json/pickle file that maps
        # task names.

        def _get_mt10_defualt_task_desc_for_env(env_name: str) -> List[str]:
            if 'goal-observable' in env_name:
                name_with_hyphen = env_name.split('-v2-goal-observable')[0]
                env_name = name_with_hyphen.replace('-', ' ')
                return [env_name]
            else:
                return [env_name]

        train_envs, heldout_envs = [], []
        for env_name in train_env_names:
            task_descriptions = _get_mt10_defualt_task_desc_for_env(env_name)
            train_envs.append(env_constructor(
                env_name=env_name,
                task_descriptions=task_descriptions,
                **common_env_kwargs,))

        for env_name in heldout_env_names:
            task_descriptions = _get_mt10_defualt_task_desc_for_env(env_name)
            heldout_envs.append(env_constructor(
                env_name=env_name,
                task_descriptions=task_descriptions,
                **common_env_kwargs,))

        return train_envs, heldout_envs


def env_constructor(env_name: str,
                    image_width: int = 256,
                    image_height: int = 256,
                    camera_name: Optional[str] = None,
                    pixel_based: bool = True,
                    render_gpu_id: int = 0,
                    proprio: int = 0,
                    lang_cond: bool = False,
                    gc=False,
                    episode_len: int = 500,
                    multi_task: bool = False,
                    task_descriptions: Optional[List[str]] = None,
                    ):
    """Create general metaworld environment with randomized goals.
    """

    if not pixel_based:
        raise ValueError("Only supports pixel based environments.")

    ## Need to do some special environment config for the metaworld environments
    if "v2" in env_name:
        e  = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
        e._freeze_rand_vec = False
        e.spec = namedtuple('spec', ['id', 'max_episode_steps'])
        e.spec.id = env_name
        e.spec.max_episode_steps = episode_len
    else:
        e = gym.make(env_name)

    e = MuJoCoDictObs(e, width=image_width, height=image_height,
                      camera_name=camera_name, device_id=render_gpu_id,
                      proprio_size=proprio, multi_task=multi_task,
                      task_name=env_name, task_descriptions=task_descriptions)
    e = GymEnv(e)

    return e


def sawyer_pick_and_place_multitask_env_constructor(
    env_name: str,
    env_type: str,
    env_config: Mapping[str, Any],
    camera_names=['left_cap2'],
    image_width: int = 256,
    image_height: int = 256,
    pixel_based: bool = True,
    render_gpu_id: int = 0,
    proprio: int = 0,
    lang_cond: bool = False,
    gc=False,
    episode_len: int = 500,
    multi_task: bool = False,):
    """Create different versions (with randomized objects) for franka pick and place env.
    """
    from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_multitask_env_mocap import SawyerPickAndPlaceMultiTaskEnvV2

    # TODO: Store task-specific keywords for a given environment so that we can
    # construct task descriptions separately without generating new data.

    # NOTE: Can format the description if we consistently know what they contain.
    if 'target' in env_type:
        task_descriptions = [env_config['task_command_type']]
    elif 'lang' in env_type:
        task_descriptions = [env_config['task_command_lang']]
    elif 'color' in env_type:
        task_descriptions = [env_config['task_command_color']]
    else:
        raise NotImplementedError("Task command type not implemented.")

    env = SawyerPickAndPlaceMultiTaskEnvV2(env_config)
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.spec = namedtuple('spec', ['id', 'max_episode_steps'])
    env.spec.id = env_name
    env.spec.max_episode_steps = episode_len

    # TODO(Mohit): Can add another environment wrapper on top of DictObs, but fine for now.
    e = MuJoCoDictObs(env, width=image_width, height=image_height,
                      camera_names=camera_names, device_id=render_gpu_id,
                      proprio_size=proprio, multi_task=multi_task,
                      task_name=env_name, task_descriptions=task_descriptions)
    e = GymEnv(e)

    return e


def parse_rlbench_env_configs_to_env_names(job_data, filter_cfg: Optional[DictConfig] = None):
    '''Parse RLBench environment config to get environments, variations to train on.'''
    env_type = job_data['rlbench_envs']['use']

    train_envs_with_variations = []
    heldout_envs_with_variations = []
    for env_name, env_configs in job_data['rlbench_envs'][env_type]['data_dirs'].items():

        for env_data_key, env_config in env_configs.items():
            data_dir = Path(env_config.data)
            assert data_dir.exists(), f"Cannot find data for rlbench env: {data_dir}"
            train_variations = env_config.train_variations

            print(f'==> Using env: {env_name}')
            print(f'\t\t Train Variations: {env_config["train_variations"]}')
            print(f'\t\t Eval Variations: {env_config["eval_variations"]}')

            for variation in env_config['train_variations']:
                data_dir_variation = data_dir / f'variation{variation}'
                assert data_dir_variation.exists(), f'Cannot find rlbench env variation data: {data_dir_variation}'
                assert (data_dir_variation / 'episodes').exists(), f'Cannot find rlbench env variation data: {data_dir_variation}'
                train_envs_with_variations.append(RLBenchEnvVariationData(env_name, variation, env_data_key, data_dir_variation / 'episodes'))
                if (data_dir_variation / 'val_episodes').exists():
                    heldout_envs_with_variations.append(RLBenchEnvVariationData(env_name, variation, env_data_key, data_dir_variation / 'val_episodes'))

    return {
        'train_env_info': train_envs_with_variations,
        'heldout_env_info': heldout_envs_with_variations,
    }


def parse_realworld_env_configs_to_env_names(job_data, filter_cfg: Optional[DictConfig] = None):
    '''Parse RLBench environment config to get environments, variations to train on.'''
    env_type = job_data['realworld_envs']['use']

    train_envs_with_variations = []
    heldout_envs_with_variations = []
    for env_name, env_configs in job_data['realworld_envs'][env_type]['data_dirs'].items():

        for env_data_key, env_config in env_configs.items():
            data_dir = Path(env_config.data)
            assert data_dir.exists(), f"Cannot find data for realworld env: {data_dir}"

            # All realworld envs have only one variation
            print(f'==> Using env: {env_name}')
            train_envs_with_variations.append(RealWorldEnvVariationData(env_name, 0, env_data_key, data_dir))
            heldout_envs_with_variations.append(RLBenchEnvVariationData(env_name, 0, env_data_key, data_dir))

    return {
        'train_env_info': train_envs_with_variations,
        'heldout_env_info': heldout_envs_with_variations,
    }


def sawyer_pick_and_place_multitask_multiview_robustness_env_constructor(
    env_name: str,
    env_type: str,
    env_config: Mapping[str, Any],
    camera_names=['left_cap2'],
    image_width: int = 256,
    image_height: int = 256,
    pixel_based: bool = True,
    render_gpu_id: int = 0,
    proprio: int = 0,
    lang_cond: bool = False,
    gc=False,
    episode_len: int = 500,
    multi_task: bool = False,
    multi_temporal_sensors: Optional[Mapping[str, Any]] = None,
    multi_step_cfg: Optional[Mapping[str, Any]] = None):
    """Create environment with procedurally added objects.
    """

    # TODO: Store task-specific keywords for a given environment so that we can
    # construct task descriptions separately without generating new data.

    # NOTE: Can format the description if we consistently know what they contain.
    if 'target' in env_type:
        task_descriptions = [env_config['task_command_type']]
    elif 'lang' in env_type:
        task_descriptions = [env_config['task_command_lang']]
    elif 'color' in env_type:
        task_descriptions = [env_config['task_command_color']]
    else:
        raise NotImplementedError("Task command type not implemented.")

    env = SawyerPickAndPlaceMultiTaskEnvProcV2(env_config,)
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.spec = namedtuple('spec', ['id', 'max_episode_steps'])
    env.spec.id = env_name
    env.spec.max_episode_steps = episode_len

    # TODO(Mohit): Can add another environment wrapper on top of DictObs, but fine for now.
    e = MuJoCoDictObs(env, width=image_width, height=image_height,
                      camera_names=camera_names, device_id=render_gpu_id,
                      proprio_size=proprio, multi_task=multi_task,
                      task_name=env_name, task_descriptions=task_descriptions,
                      multi_temporal_sensors=multi_temporal_sensors,
                      proprio_key='ee_xyz')
    if multi_step_cfg is not None and multi_step_cfg['use']:
        e = MultiStepWrapper(
            e,
            multi_step_cfg['n_obs_steps'],
            multi_step_cfg['n_action_steps'],
            max_episode_steps=episode_len, reward_agg_method='sum')
            # e, multi_step_cfg['n_obs_steps'], multi_step_cfg['n_action_steps'])
    e = GymEnv(e)

    return e


def sawyer_pick_and_place_multitask_multiskill_env_constructor(
    env_name: str,
    env_type: str,
    env_config: Mapping[str, Any],
    camera_names=['left_cap2'],
    image_width: int = 256,
    image_height: int = 256,
    pixel_based: bool = True,
    render_gpu_id: int = 0,
    proprio: int = 0,
    lang_cond: bool = False,
    gc=False,
    episode_len: int = 500,
    multi_task: bool = False,
    multi_temporal_sensors: Optional[Mapping[str, Any]] = None,
    multi_step_cfg: Optional[Mapping[str, Any]] = None):
    """Create different versions (with randomized objects) for franka pick and place env.
    """
    from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_multitask_env_gen_mocap import SawyerPickAndPlaceMultiTaskGenEnvV2

    # TODO: Store task-specific keywords for a given environment so that we can
    # construct task descriptions separately without generating new data.

    # NOTE: Can format the description if we consistently know what they contain.
    if 'target' in env_type:
        task_descriptions = [env_config['task_command_type']]
    elif 'lang' in env_type:
        task_descriptions = [env_config['task_command_lang']]
    elif 'color' in env_type:
        task_descriptions = [env_config['task_command_color']]
    else:
        raise NotImplementedError("Task command type not implemented.")

    env = SawyerPickAndPlaceMultiTaskGenEnvV2(env_config,)
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.spec = namedtuple('spec', ['id', 'max_episode_steps'])
    env.spec.id = env_name
    env.spec.max_episode_steps = episode_len

    # TODO(Mohit): Can add another environment wrapper on top of DictObs, but fine for now.
    e = MuJoCoDictObs(env, width=image_width, height=image_height,
                      camera_names=camera_names, device_id=render_gpu_id,
                      proprio_size=proprio, multi_task=multi_task,
                      task_name=env_name, task_descriptions=task_descriptions,
                      multi_temporal_sensors=multi_temporal_sensors)
    if multi_step_cfg is not None and multi_step_cfg['use']:
        e = MultiStepWrapper(
            e, 2, 8, max_episode_steps=episode_len, reward_agg_method='sum')
            # e, multi_step_cfg['n_obs_steps'], multi_step_cfg['n_action_steps'])
    e = GymEnv(e)

    return e


def sawyer_multiobj_multitask_env_constructor(
    env_name: str,
    env_type: str,
    env_config: Mapping[str, Any],
    camera_names=['left_cap2'],
    image_width: int = 256,
    image_height: int = 256,
    pixel_based: bool = True,
    render_gpu_id: int = 0,
    proprio: int = 0,
    lang_cond: bool = False,
    gc=False,
    episode_len: int = 500,
    multi_task: bool = False,):
    """Create different versions (with randomized objects) for franka pick and place env.
    """

    # TODO: Store task-specific keywords for a given environment so that we can
    # construct task descriptions separately without generating new data.

    # NOTE: Can format the description if we consistently know what they contain.
    if 'target' in env_type:
        # We have multiple task commands for MultiObjMultiSkill data with sticks.
        # Hence, store all of them.
        if (isinstance(env_config['task_command_type'], list) or
            isinstance(env_config['task_command_type'], ListConfig)):
            task_descriptions = [desc for desc in env_config['task_command_type']]
        else:
            task_descriptions = [env_config['task_command_type']]
    elif 'lang' in env_type:
        task_descriptions = [env_config['task_command_lang']]
    elif 'color' in env_type:
        task_descriptions = [env_config['task_command_color']]
    else:
        raise NotImplementedError("Task command type not implemented.")

    if env_config.get('has_small_objects', False):
        env = SawyerMultiObjectMultiTaskMultiShapeGenEnvV2(env_config, data_collection=False,)
    else:
        env = SawyerMultiObjectMultiTaskGenEnvV2(env_config, data_collection=False,)
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.spec = namedtuple('spec', ['id', 'max_episode_steps'])
    env.spec.id = env_name
    env.spec.max_episode_steps = episode_len

    # TODO(Mohit): Can add another environment wrapper on top of DictObs, but fine for now.
    e = MuJoCoDictObs(env, width=image_width, height=image_height,
                      camera_names=camera_names, device_id=render_gpu_id,
                      proprio_size=proprio, multi_task=multi_task,
                      task_name=env_name, task_descriptions=task_descriptions,
                      proprio_key='ee_xyz')
    e = GymEnv(e)

    return e


def ballbot_pickup_multitask_multiview_env_constructor(
    env_name: str,
    env_type: str,
    env_config: Mapping[str, Any],
    camera_names=['left_cap2'],
    image_width: int = 256,
    image_height: int = 256,
    pixel_based: bool = True,
    render_gpu_id: int = 0,
    proprio: int = 0,
    lang_cond: bool = False,
    gc=False,
    episode_len: int = 500,
    multi_task: bool = False,
    multi_temporal_sensors: Optional[Mapping[str, Any]] = None,
    min_action = None,
    max_action = None,
    action_type = 'delta_obs_pos',
    all_low_res = False,
    all_high_res = False):
    """Create different versions (with randomized objects) for franka pick and place env.
    """

    # TODO: Store task-specific keywords for a given environment so that we can
    # construct task descriptions separately without generating new data.

    # NOTE: Can format the description if we consistently know what they contain.
    if 'target' in env_type:
        task_descriptions = [env_config['task_command_type']]
    elif 'lang' in env_type:
        task_descriptions = [env_config['task_command_lang']]
    elif 'color' in env_type:
        task_descriptions = [env_config['task_command_color']]
    else:
        raise NotImplementedError("Task command type not implemented.")
    
    env = RobotSimulatorPickup(None,env_config)

    env.spec = namedtuple('spec', ['id', 'max_episode_steps', 'observation_dim', 'action_dim', 'horizon'])
    env.spec.id = env_name
    env.spec.max_episode_steps = episode_len
    env.spec.observation_dim = 4
    env.spec.action_dim = 4

    I3_save_freq = env_config['I3_save_freq']
    Ih_save_freq = env_config['Ih_save_freq']
    FT_save_freq = env_config['FT_save_freq']
    if multi_temporal_sensors is not None:
        if all_low_res:
            multi_temporal_sensors.i3_freq = int(I3_save_freq/FT_save_freq)
            multi_temporal_sensors.ih_freq = int(I3_save_freq/FT_save_freq) # Ih made slower
        elif all_high_res:
            multi_temporal_sensors.i3_freq = int(I3_save_freq/FT_save_freq) # I3 should not be made faster? as it is not faster in the dataset
            multi_temporal_sensors.ih_freq = int(Ih_save_freq/FT_save_freq)
        else:
            multi_temporal_sensors.i3_freq = int(48/FT_save_freq)
            multi_temporal_sensors.ih_freq = int(48/FT_save_freq) # TODO(saumya): remove hardcoded

    e = BallbotDictObs(env, width=image_width, height=image_height,
                      camera_names=camera_names, device_id=render_gpu_id,
                      proprio_size=proprio, multi_task=multi_task,
                      task_name=env_name, task_descriptions=task_descriptions,
                      multi_temporal_sensors=multi_temporal_sensors,
                      min_action = min_action,
                      max_action = max_action,
                      action_type = action_type,
                      all_low_res=all_low_res,
                      all_high_res=all_high_res)
    e.horizon = episode_len
    e.env_id = env_name

    return e

