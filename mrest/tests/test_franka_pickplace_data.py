import os
import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
from mrest.utils.sampling import do_rollout
from omegaconf import OmegaConf
from pandas import DataFrame
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from metaworld.envs.mujoco.franka_pickandplace.franka_pick_place_multitask_env import FrankaPickPlaceMultitaskEnv
from mrest.data_gen.abr_policies.pick_place_mj_abr_policy import PickPlaceMjAbrPolicy
from mrest.utils.env_helpers import franka_pick_and_place_multitask_env_constructor

def plot_action_hist(data_dir=''):
    all_actions = []
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    for env_idx, env_dir in enumerate(data_dir.iterdir()):
        for demo_idx, demo_dir in enumerate(env_dir.iterdir()):
            with open(f'{demo_dir}/info.pickle', 'rb') as info:
                info = pickle.load(info)
                actions = info['actions_des_delta']
            all_actions.append(actions)
    
    all_actions = np.concatenate(all_actions, axis=0)
    print('all_actions shape',all_actions.shape)
    df = DataFrame({
        # 'actions0': all_actions[:,0],
        # 'actions1': all_actions[:,1],
        'actions2': all_actions[:,2],
        # 'actions3': all_actions[:,3],
        })
    fig = df.plot.hist().get_figure()
    fig.savefig(f'./media/actions_hist.jpg')

def plot_action_traj(data_dir=''):
    with open(f'{data_dir}/info.pickle', 'rb') as info:
        info = pickle.load(info)
        actions = info['actions_des_delta']
        obs = info['observations']
    des_obs = obs[:, :3] + actions[:, :3]
    df = DataFrame({
        # 'des_obs0': des_obs[:,0],
        # 'des_obs1': des_obs[:,1],
        # 'des_obs2': des_obs[:,2],
        # 'obs0': obs[:,0],
        # 'obs1': obs[:,1],
        # 'obs2': obs[:,2],
        'actions0': actions[:,0],
        'actions1': actions[:,1],
        'actions2': actions[:,2],
        })
    fig = df.plot().get_figure()
    fig.savefig(f'./media/actions_plot_des_delta.jpg')

def main():

    if True:
        demo_dir = "/home/saumyas/experiment_results/object_centric/r3m/data/ballbot/ballbot_pickup_fast_multires_PO_objloc/"
        data_type = 'train/'
        env_name = "env_ballbot_pick_red_block/"
        demo_idx = 1
        camera_name = 'eye_in_hand_90'
        camera_name = 'left_cap2'

        img_dir = demo_dir + data_type + env_name + f'demo{demo_idx}/' + camera_name
        data_dir = demo_dir + data_type + env_name + f'demo{demo_idx}/'
        imgs = []
        demo_images = [f for f in Path(img_dir).iterdir()]
        demo_images = sorted(demo_images, key=lambda x: int(x.name.split('img_t')[1].split('.')[0]))
        for demo_img in demo_images:
            if os.path.isfile(demo_img):
                img = np.asarray(Image.open(demo_img))
                imgs.append(img[:,:,:3])
            else:
                break
        
        print(f"Saving file of len:{len(demo_images)}")
        filename = './media/ballbot_I3.gif'
        cl = ImageSequenceClip(imgs, fps=50)
        cl.write_gif(filename, fps=50, logger=None)
        del cl

        plot_action_hist(data_dir=(demo_dir + data_type))
        plot_action_traj(data_dir=data_dir)

    if False:
        env_cfg = {"target_object": "block",
                     "skill": "Pick",
                     "task_command_type": "Pick up block",
                     "task_command_color": "Pick up red object",
                     "task_command_lang": "Pick up red object"}
        job_data = OmegaConf.load('/home/saumyas/Projects/ms_r3m/evaluation/mrest/core/config/BC_train_multitask_config.yaml')
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
            'episode_len': 250,
            'multi_task': True,
        }
        env = franka_pick_and_place_multitask_env_constructor(
                    env_name='env_target_block',
                    env_type='mt_franka_pick_target',
                    env_config=env_cfg,
                    **common_env_kwargs)
        # env = FrankaPickPlaceMultitaskEnv(env_cfg)
        policy = PickPlaceMjAbrPolicy(env.env.env.robot_config, skill=env_cfg['skill'], target_object=env_cfg['target_object'])

        paths = do_rollout(
            1,
            env,
            policy,
            eval_mode = False,
            horizon = 250,
            base_seed = None,
            env_kwargs = None,
            env_has_image_embedding_wrapper=False)

        imgs = paths[0]['images']
        filename = 'rollout.gif'
        cl = ImageSequenceClip(imgs, fps=50)
        cl.write_gif(filename, fps=50, logger=None)
        del cl
if __name__=="__main__":
    main()