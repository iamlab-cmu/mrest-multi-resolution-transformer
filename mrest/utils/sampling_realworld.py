# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import numpy as np
import time
from mrest.utils import tensor_utils
logging.disable(logging.CRITICAL)
import multiprocessing as mp
import time as timer
logging.disable(logging.CRITICAL)
import gc
from collections import namedtuple

from mrest.utils.mdetr.mdetr_object_detection import object_mask_mdetr


# Single core rollout to sample trajectories
# =======================================================
def do_realworld_rollout(
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
        append_object_mask=None,
        obj_detection_model=None,
        obj_detection_processor=None,
        do_dynamic_motion: bool = False,
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

        horizon = 6500

        while t < horizon and done != True:
            if t == 0 and append_object_mask is not None:
                if append_object_mask == 'mdetr':
                    object_mask = object_mask_mdetr(obj_detection_model, o['left_cap2'], env.env.task_name, env.env.task_descriptions[0])
            
            if append_object_mask is not None:
                o['left_cap2'] = np.append(o['left_cap2'], object_mask[:,:,None], axis=2)
            
            inference_start_time = time.time()
            a, agent_info = policy.get_action(o)
            inference_end_time = time.time()

            print(f"Inference time: {inference_end_time - inference_start_time:.6f}")
            assert eval_mode, "Use eval mode only for now"
            if eval_mode:
                a = agent_info['evaluation']

            next_o, r, done, env_info_step = env.step(a)
            env_info = env_info_step #if env_info_base == {} else env_info_base

            observations.append(o)
            actions.append(a)
            rewards.append(r)
            try:
                if hasattr(env.env, 'camera_names'):
                    # For two cameras just concatenate the images and save one gif (easy for visualization)
                    if len(env.env.camera_names) == 2:
                        if img_by_camera_name.get(cam0 + '-' + cam1) is None:
                            img_by_camera_name[cam0 + '-' + cam1] = []
                        img_by_camera_name[cam0 + '-' + cam1].append(np.hstack([o[cam0], o[cam1]]))
                    else:
                        for cam_name in env.env.camera_names:
                            if img_by_camera_name.get(cam_name) is None:
                                img_by_camera_name[cam_name] = []
                            img_by_camera_name[cam_name].append(o[cam_name])

                if env_has_image_embedding_wrapper:
                    ims.append(env.env.env.get_image())
                else:
                    ims.append(env.env.get_image())
            except:
                pass
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = next_o
            t += 1
        
        env.close()
        breakpoint()

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done, 
            images=ims,
            images_by_camera_name=img_by_camera_name,
        )

        paths.append(path)
        ep += 1
    
    if append_object_mask == 'mdetr':
        del obj_detection_model
    
    if append_object_mask == 'owl_vit':
        del obj_detection_model
        del obj_detection_processor

    # del(env)
    del env

    gc.collect()
    return paths


def do_dynamic_realworld_rollout(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        env_has_image_embedding_wrapper: bool = True,
        append_object_mask=None,
        obj_detection_model=None,
        obj_detection_processor=None,
        **kwargs,
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

    env = env
    ep = 0
    while ep < num_traj:
        env.do_dynamic_episode_rollout(policy)
        breakpoint()

    return {}