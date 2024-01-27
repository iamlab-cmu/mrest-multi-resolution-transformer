# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import numpy as np
from mrest.utils.gym_env import GymEnv
from mrest.utils import tensor_utils
logging.disable(logging.CRITICAL)
import multiprocessing as mp
import time as timer
logging.disable(logging.CRITICAL)
import gc
from collections import namedtuple
import rlbench

from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)


# Single core rollout to sample trajectories
# =======================================================
def do_rlbench_rollout(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        env_kwargs = None,
        env_has_image_embedding_wrapper: bool = True,
        set_seed_every_episode: bool = True,
        set_numpy_seed_per_env: bool = True,
):
    """
    :param num_traj:    number of trajectories (int)
    :param env:         environment (env class, str with env_name, or factory function)
    :param policy:      policy to use for action selection
    :param eval_mode:   use evaluation mode for action computation (bool)
    :param horizon:     max horizon length for rollout (<= env.horizon)
    :param base_seed:   base seed for rollouts (int)
    :param env_kwargs:  dictionary with parameters, will be passed to env generator
    :param set_seed_every_episode: reset seed ever episode
    :param set_numpy_seed_per_env: Should we set global numpy seed per env.
                                   NOTE: by default env seed is always set.
    :return:
    """
    # get the correct env behavior
    if type(env) == str:
        raise NotImplementedError
    elif callable(env):
        raise NotImplementedError
    else:
        env = env

    if base_seed is not None and set_numpy_seed_per_env:
        np.random.seed(base_seed)
    # horizon = min(horizon, env.horizon)
    paths = []

    ep = 0
    while ep < num_traj:

        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []

        o = env.reset()
        done = False
        t = 0
        ims = []
        img_by_camera_name = {}

        if hasattr(env, 'camera_names') and len(env.camera_names) == 2:
            cam0, cam1 = env.camera_names
        elif hasattr(env, 'camera_names') and len(env.camera_names) == 1:
            cam0 = env.camera_names[0]

        ## MetaWorld vs. Adroit/Kitchen syntax
        # init_state = env.__getstate__()
        horizon = 60

        print("==== Begin episode ====")
        while t < horizon and done != True:
            a, agent_info = policy.get_action(o)
            assert eval_mode, "Use eval mode only for now"
            if eval_mode:
                a = agent_info['evaluation']

            try:
                next_o, r, done, env_info_step = env.step(a)
                env_info = env_info_step #if env_info_base == {} else env_info_base
                observations.append(o)
                actions.append(a)
                rewards.append(r)
                if hasattr(env, 'camera_names'):
                    # For two cameras just concatenate the images and save one gif (easy for visualization)
                    if len(env.camera_names) == 2:
                        if img_by_camera_name.get(cam0 + '-' + cam1) is None:
                            img_by_camera_name[cam0 + '-' + cam1] = []
                        img_by_camera_name[cam0 + '-' + cam1].append(np.hstack([o[cam0], o[cam1]]))
                    else:
                        for cam_name in env.camera_names:
                            if img_by_camera_name.get(cam_name) is None:
                                img_by_camera_name[cam_name] = []
                            img_by_camera_name[cam_name].append(o[cam_name])

                agent_infos.append(agent_info)
                env_infos.append(env_info)
                o = next_o
                t += 1

            except rlbench.backend.exceptions.InvalidActionError as e:
                print("Cannot exec action RLBench Invalid action", e)
                break


        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done,
            images=img_by_camera_name[cam0] if len(env.camera_names) == 1 else None,
            images_by_camera_name=img_by_camera_name,
        )

        paths.append(path)
        ep += 1

    del(env)
    gc.collect()
    return paths


def _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts):

    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=None)
    parallel_runs = [pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        pool.terminate()
        pool.join()
        return _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts-1)

    pool.close()
    pool.terminate()
    pool.join()
    return results
