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

from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
from mrest.utils.mdetr.mdetr_object_detection import object_mask_mdetr

# Single core rollout to sample trajectories
# =======================================================
def do_rollout(
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
        obj_detection_processor=None
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
        ## MetaWorld specific stuff
        if "v2" in env:
            env_name = env
            env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
            env._freeze_rand_vec = False
            env.horizon = 500
            env.spec = namedtuple('spec', ['id', 'max_episode_steps', 'observation_dim', 'action_dim'])
            env.spec.id = env_name
            env.spec.observation_dim = int(env.observation_space.shape[0])
            env.spec.action_dim = int(env.action_space.shape[0])
            env.spec.max_episode_steps = 500
        else:
            env = GymEnv(env)
    elif isinstance(env, GymEnv):
        env = env
    elif callable(env):
        env = env(**env_kwargs)
    else:
        # print("Unsupported environment format")
        # raise AttributeError
        ## Support passing in one env for everything
        env = env

    if base_seed is not None:
        try:
            env.set_seed(base_seed)
        except:
            env.seed(base_seed)
        if set_numpy_seed_per_env:
            np.random.seed(base_seed)
    else:
        raise NotImplementedError
    # horizon = min(horizon, env.horizon)
    paths = []

    ep = 0
    while ep < num_traj:
        print(f"Starting episode: {ep}")
        # seeding
        if base_seed is not None and set_seed_every_episode:
            seed = base_seed + ep
            try:
                env.set_seed(seed)
            except:
                env.seed(seed)
            np.random.seed(seed)

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

        if hasattr(env.env, 'camera_names') and len(env.env.camera_names) == 2:
            cam0, cam1 = env.env.camera_names

        # try:
        #     if env_has_image_embedding_wrapper:
        #         img = env.env.env.get_image()
        #     else:
        #         img = env.env.get_image()
        #     ims.append(img)
        # except:
        #     ## For state based learning
        #     pass

        ## MetaWorld vs. Adroit/Kitchen syntax
        try:
            init_state = env.__getstate__()
        except:
            init_state = env.get_env_state()

        while t < horizon and done != True:
            if t == 0 and append_object_mask is not None:
                if append_object_mask == 'mdetr':
                    object_mask = object_mask_mdetr(obj_detection_model, o['left_cap2'], env.env.task_name, env.env.task_descriptions[0])
            
            if append_object_mask is not None:
                o['left_cap2'] = np.append(o['left_cap2'], object_mask[:,:,None], axis=2)
            
            a, agent_info = policy.get_action(o)
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

                # if env_has_image_embedding_wrapper:
                #     ims.append(env.env.env.get_image())
                # else:
                #     ims.append(env.env.get_image())
            except:
                pass
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = next_o
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done, 
            init_state = init_state,
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


def sample_paths(
        num_traj,
        env,
        policy,
        eval_mode = False,
        horizon = 1e6,
        base_seed = None,
        num_cpu = 1,
        max_process_time=300,
        max_timeouts=4,
        suppress_print=False,
        env_kwargs=None,
        env_has_image_embedding_wrapper: bool = True,
        set_numpy_seed_per_env: bool = True,
        append_object_mask=None,
        obj_detection_model=None,
        obj_detection_processor=None,
        env_type: str = 'metaworld',
        ):

    num_cpu = 1 if num_cpu is None else num_cpu
    # num_cpu = mp.cpu_count() if num_cpu == 'max' else num_cpu
    assert type(num_cpu) == int


    if env_type == 'rlbench':
        from mrest.utils.sampling_rlbench import do_rlbench_rollout
        input_dict = dict(num_traj=num_traj, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon, base_seed=base_seed,
                          env_kwargs=env_kwargs, 
                          env_has_image_embedding_wrapper=env_has_image_embedding_wrapper,
                          set_numpy_seed_per_env=set_numpy_seed_per_env)
        return do_rlbench_rollout(**input_dict)
    elif env_type == 'realworld':
        from mrest.utils.sampling_realworld import do_realworld_rollout, do_dynamic_realworld_rollout
        input_dict = dict(num_traj=num_traj, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon, base_seed=base_seed,
                          env_kwargs=env_kwargs, 
                          env_has_image_embedding_wrapper=env_has_image_embedding_wrapper,
                          set_numpy_seed_per_env=set_numpy_seed_per_env)
        # return do_realworld_rollout(**input_dict)
        return do_dynamic_realworld_rollout(**input_dict)

    if env_type == 'pybullet':
        from mrest.utils.sampling_pybullet import do_pybullet_rollout
        input_dict = dict(num_traj=num_traj, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon, base_seed=base_seed,
                          env_kwargs=env_kwargs, 
                          env_has_image_embedding_wrapper=env_has_image_embedding_wrapper,
                          set_numpy_seed_per_env=set_numpy_seed_per_env)
        return do_pybullet_rollout(**input_dict)
    
    if num_cpu == 1:
        input_dict = dict(num_traj=num_traj, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon, base_seed=base_seed,
                          env_kwargs=env_kwargs, 
                          env_has_image_embedding_wrapper=env_has_image_embedding_wrapper,
                          set_numpy_seed_per_env=set_numpy_seed_per_env,
                          append_object_mask=append_object_mask,
                          obj_detection_model=obj_detection_model,
                          obj_detection_processor=obj_detection_processor)
        # dont invoke multiprocessing if not necessary
        return do_rollout(**input_dict)

    # do multiprocessing otherwise
    paths_per_cpu = int(np.ceil(num_traj/num_cpu))
    input_dict_list= []
    for i in range(num_cpu):
        input_dict = dict(num_traj=paths_per_cpu, env=env, policy=policy,
                          eval_mode=eval_mode, horizon=horizon,
                          base_seed=base_seed + i * paths_per_cpu,
                          env_kwargs=env_kwargs,
                          env_has_image_embedding_wrapper=env_has_image_embedding_wrapper)
        input_dict_list.append(input_dict)
    if suppress_print is False:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    results = _try_multiprocess(do_rollout, input_dict_list,
                                num_cpu, max_process_time, max_timeouts)
    paths = []
    # result is a paths type and results is list of paths
    for result in results:
        for path in result:
            paths.append(path)  

    if suppress_print is False:
        print("======= Samples Gathered  ======= | >>>> Time taken = %f " %(timer.time()-start_time) )
    
    del env
    if append_object_mask == 'mdetr':
        del obj_detection_model
    
    if append_object_mask == 'owl_vit':
        del obj_detection_model
        del obj_detection_processor
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
