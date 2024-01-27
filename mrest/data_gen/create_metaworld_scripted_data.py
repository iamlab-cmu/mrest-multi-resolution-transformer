import cv2
import numpy as np
import argparse
from pathlib import Path
import pickle

from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS, ALL_V2_ENVIRONMENTS
from metaworld.policies import *
# from tests.metaworld.envs.mujoco.sawyer_xyz.utils import trajectory_summary

from typing import List, Dict

ALL_ENVS = {**ALL_V1_ENVIRONMENTS, **ALL_V2_ENVIRONMENTS}


def trajectory_summary(env, policy, act_noise_pct, render=False, end_on_success=True, demo_idx=None, 
                       end_on_success_followup_steps: int = 0,
                       use_opencv_to_render: bool = False, camera_names=['left_cap2']):
    """Tests whether a given policy solves an environment
    Args:
        env (metaworld.envs.MujocoEnv): Environment to test
        policy (metaworld.policies.policies.Policy): Policy that's supposed to
            succeed in env
        act_noise_pct (np.ndarray): Decimal value(s) indicating std deviation of
            the noise as a % of action space
        render (bool): Whether to render the env in a GUI
        end_on_success (bool): Whether to stop stepping after first success
        end_on_success_followup_steps: (int): Steps to wait after completion for the episode to end.
        use_opencv_to_render: (bool): Wether render the image being used as observation using OpenCV.
    Returns:
        (bool, np.ndarray, np.ndarray, int): Success flag, Rewards, Returns,
            Index of first success
    """
    success = False
    first_success = 0

    observations = []
    actions = []
    actions_with_noise = []
    rewards = []
    env_infos = []
    frames = {camera_name: [] for camera_name in camera_names}

    for t, (o, a, a_noise, r, done, info, frame, next_obs) in enumerate(trajectory_generator(
        env, policy, act_noise_pct, render, use_opencv_to_render=use_opencv_to_render, camera_names=camera_names)):
        rewards.append(r)
        # assert not env.isV2 or set(info.keys()) == {
        #     'success',
        #     'near_object',
        #     'grasp_success',
        #     'grasp_reward',
        #     'in_place_reward',
        #     'obj_to_target',
        #     'unscaled_reward'
        # }
        success |= bool(info['success'])
        if not success:
            first_success = t
        
        if isinstance(o, dict):
            observations.append(np.r_[o['ee_xyz'], o['goal']])
        else:
            observations.append(o)
        actions.append(a)
        actions_with_noise.append(a_noise)
        rewards.append(r)
        env_infos.append(info)
        
        for camera_name in camera_names:
            frames[camera_name].append(frame[camera_name])

        if (success or done) and end_on_success:
            if end_on_success_followup_steps == 0:
                break
            if end_on_success_followup_steps > 0 and (t - first_success) >= end_on_success_followup_steps:
                break
    if success:
        print(f"First success: {first_success}")
    else:
        print(f"No success")
    
    # ==== Uncomment to debug ====
    # if frames is not None and len(frames) > 0:
    #     if len(frames) > 0:
    #         import os
    #         import imageio
    #         if demo_idx is not None:
    #             # path = os.path.join('/home/mohit/assembly_frames/test_0.035', f'run_{demo_idx:04}.gif')
    #             path = os.path.join('/home/mohit/button_press_frames/test_0.035', f'run_{demo_idx:04}.gif')
    #         else:
    #             # path = os.path.join('/home/mohit/assembly_frames/', 'run_2.gif')
    #             path = os.path.join('/home/mohit/button_press_frames/', 'run_2.gif')
    #         imageio.mimsave(path, frames, fps=20)


    rewards = np.array(rewards)
    returns = np.cumsum(rewards)

    # Create the trajectory dict?
    demo_dict = {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'actions_with_noise': np.array(actions_with_noise),
        'rewards': np.array(rewards),
        'env_infos': env_infos,
    }
    for camera_name in camera_names:
        demo_dict.update({camera_name: np.array(frames[camera_name])})
    
    return success, rewards, returns, first_success, demo_dict


def trajectory_generator(env, policy, act_noise_pct, render=False, use_opencv_to_render: bool = False, camera_names=['left_cap2']):
    """Tests whether a given policy solves an environment
    Args:
        env (metaworld.envs.MujocoEnv): Environment to test
        policy (metaworld.policies.policies.Policy): Policy that's supposed to
            succeed in env
        act_noise_pct (np.ndarray): Decimal value(s) indicating std deviation of
            the noise as a % of action space
        render (bool): Whether to render the env in a GUI
    Yields:
        (float, bool, dict): Reward, Done flag, Info dictionary
    """
    action_space_ptp = env.action_space.high - env.action_space.low

    env.reset()
    env.reset_model()
    o = env.reset()
    # assert o.shape == env.observation_space.shape
    # assert env.observation_space.contains(o), obs_space_error_text(env, o)

    if use_opencv_to_render:
        cv2.namedWindow(f"{type(env).__name__}", cv2.WINDOW_AUTOSIZE)

    for step in range(env.max_path_length):
        a = policy.get_action(o)
        a_noise = np.random.normal(a, act_noise_pct * action_space_ptp)

        next_obs, r, done, info = env.step(a_noise)
        # assert env.observation_space.contains(o), obs_space_error_text(env, next_obs)
        if render:
            # env.render()
            # top_cap2, left_cap2, right_cap2
            frame = {}
            for camera_name in camera_names:
                try:
                    _img = env.sim.render(
                        # mode='rgb_array',
                        height=256,
                        width=256,
                        camera_name=camera_name,
                    )
                    frame[camera_name] = _img[::-1, :, :]
                except:
                    _img = env.render(offscreen=True, camera_name=camera_name, resolution=(256, 256))
                    frame[camera_name] = _img.copy()

            if use_opencv_to_render:
                img = cv2.cvtColor(frame['left_cap2'], cv2.COLOR_BGR2RGB)
                cv2.imshow(f'{type(env).__name__}', img)
                cv2.waitKey(1)

        yield o, a, a_noise, r, done, info, frame, next_obs
        o = next_obs
    print("Episode done")


def obs_space_error_text(env, obs):
    return "Obs Out of Bounds\n\tlow: {}, \n\tobs: {}, \n\thigh: {}".format(
        env.observation_space.low[[0, 1, 2, -3, -2, -1]],
        obs[[0, 1, 2, -3, -2, -1]],
        env.observation_space.high[[0, 1, 2, -3, -2, -1]]
    )


def reduce_demo_dicts_to_dict(demo_dicts: List[Dict]) -> Dict:
    if len(demo_dicts) == 0:
        return {}
    
    dict_keys = list(demo_dicts[0].keys())
    all_demo_dict = {}
    # convert to arrays
    for k in dict_keys:
        all_demo_dict[k] = []
        for i, demo_dict in enumerate(demo_dicts):
            all_demo_dict[k].append(demo_dict[k])
        
    to_array_keys = ['observations', 'actions', 'rewards', 'images']
    for k in to_array_keys:
        all_demo_dict[k] = np.array(all_demo_dict[k])
        
    return all_demo_dict


def get_policy_for_env(env_name: str):
    if "assembly-v2" in env_name:
        policy = SawyerAssemblyV2Policy()
    elif "button-press-v2-goal-observable" in env_name: 
        policy = SawyerButtonPressV2Policy()
    elif "button-press-topdown-v2" in env_name:
        policy = SawyerButtonPressTopdownV2Policy()
    elif "button-press-topdown-wall-v2" in env_name:
        policy = SawyerButtonPressTopdownWallV2Policy()
    elif "button-press-wall-v2" in env_name:
        policy = SawyerButtonPressWallV2Policy()
    elif "reach-v2" in env_name:
        policy = SawyerReachV2Policy()
    elif "push-v2" in env_name:
        policy = SawyerPushV2Policy()
    elif "pick-place-v2" in env_name:
        policy = SawyerPickPlaceV2Policy()
    elif "door-open-v2" in env_name:
        policy = SawyerDoorOpenV2Policy()
    elif "drawer-open-v2" in env_name:
        policy = SawyerDrawerOpenV2Policy()
    elif "drawer-close-v2" in env_name:
        policy = SawyerDrawerCloseV2Policy()
    elif "peg-insert-side-v2" in env_name:
        policy = SawyerPegInsertionSideV2Policy()
    elif "window-open-v2" in env_name:
        policy = SawyerWindowOpenV2Policy()
    elif "window-close-v2" in env_name:
        policy = SawyerWindowCloseV2Policy()
    ## Beyond these are MT-20 or MT-50 tasks
    elif "basketball-v2" in env_name:
        policy = SawyerBasketballV2Policy()
    elif "coffee-button-v2" in env_name:
        policy = SawyerCoffeeButtonV2Policy()
    elif "coffee-pull-v2" in env_name:
        policy = SawyerCoffeePullV2Policy()
    elif "coffee-push-v2" in env_name:
        policy = SawyerCoffeePushV2Policy()
    elif "dial-turn-v2" in env_name:
        policy = SawyerDialTurnV2Policy()
    elif "disassemble-v2" in env_name:
        policy = SawyerDisassembleV2Policy()
    elif "door-close-v2" in env_name:
        policy = SawyerDoorCloseV2Policy()
    elif "door-lock-v2" in env_name:
        policy = SawyerDoorLockV2Policy()
    elif "door-unlock-v2" in env_name:
        policy = SawyerDoorUnlockV2Policy()
    elif "hand-insert-v2" in env_name:
        policy = SawyerHandInsertV2Policy()
    elif "faucet-open-v2" in env_name:
        policy = SawyerFaucetOpenV2Policy()
    elif "faucet-close-v2" in env_name:
        policy = SawyerFaucetCloseV2Policy()
    else:
        raise ValueError(f"Invalid env name: {env_name}")

    return policy
    

def main(args):
    # env_name = "assembly-v2"
    # env_name = "window-close-v2"
    env_name = args.env_name
    # Create the env
    env = ALL_ENVS[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    all_demo_dicts = []

    save_path = None
    if args.save_path is not None and len(args.save_path) > 0:
        save_path = Path(args.save_path)
        assert save_path.parent.exists(), f"Path does not exist: {save_path.parent}"

    # Now run the policy?
    # ['assembly-v2', SawyerAssemblyV2Policy(), .0, 1.],
    policy = get_policy_for_env(env_name)
    action_noisy_percent = 0.0
    successes = 0
    for demo_idx in range(args.num_demos):
        traj_success = 0
        while not traj_success:
            traj_info = trajectory_summary(env, policy, action_noisy_percent, render=True, end_on_success=False, demo_idx=demo_idx)
            traj_success = int(traj_info[0])
        successes += float(traj_info[0])
        demo_dict = traj_info[4]
        assert isinstance(demo_dict, dict), f"Not a valid demo dict: {type(demo_dict)}"

        all_demo_dicts.append(demo_dict)

    print(f"Total success: {successes:.4f}")
    # assert successes >= expected_success_rate * iters

    # Reduce demo dicts
    # combined_demos_dict = reduce_demo_dicts(all_demo_dicts)

    if save_path is not None:
        with open(save_path, 'wb') as pkl_f:
            pickle.dump(all_demo_dicts, pkl_f, protocol=4)
        print(f"Did save demos at: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate demo data for a given task (R3M).')
    parser.add_argument('--env_name', type=str, default='window-close-v2', help='Env to use')
    parser.add_argument('--save_path', type=str, default='', help='Path to save demo data in a pickle file.')
    parser.add_argument('--num_demos', type=int, default=4, help='Number of demos to run')
    args = parser.parse_args()

    main(args)
