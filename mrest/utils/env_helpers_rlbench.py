from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from pyrep.const import RenderMode, ObjectType
import os
import gym
import numpy as np
from collections import deque
from gym import spaces
from gym.spaces.box import Box

from rlbench import CameraConfig, ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper, MoveArmPosOnlyThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as rlbench_task
from rlbench.backend.exceptions import BoundaryError, WaypointError, TaskEnvironmentError

from mrest.utils.torch_utils import check_config_flag
from mrest.utils.gym_env import EnvSpec
from mrest.gym_utils.multistep_wrapper import MultiStepWrapper


class RLBenchDictObs(gym.ObservationWrapper):
    def __init__(self, env, task, width, height, camera_names,
                 proprio_size: int = 0,
                 horizon: int = 200,
                 task_name: Optional[str] = None,
                 task_variation: int = -1,
                 task_descriptions: Optional[List[str]] = None,
                 proprio_key: Optional[str] = None,
                 use_low_state_as_proprio: bool = False,
                 add_one_hot_timestep_config: Optional[Mapping[str, Any]] = None,
                 tanh_action_cfg: Optional[Mapping[str, Any]] = None,
                 multi_temporal_sensors: Optional[Mapping[str, Any]] = None,
                 *args, **kwargs):
        # gym.ObservationWrapper.__init__(self, env)
        self.env = env
        self.task = task

        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_names = camera_names
        self.proprio_size = proprio_size
        self.horizon = horizon
        self.use_low_state_as_proprio = use_low_state_as_proprio
        self.add_one_hot_timestep_config = add_one_hot_timestep_config
        self.multi_temporal_sensors = multi_temporal_sensors
        if multi_temporal_sensors is None or not multi_temporal_sensors['use']:
            self.multi_temporal_sensors = None
        else:
            # TODO: Fix this logic, where we don't explicitly add multi_temporal_sensor frequency.
            self.last_img_obs_dict = dict()
            for cam_name in camera_names:
                if not cam_name.endswith('_rgb'):
                    cam_name = f'{cam_name}_rgb'
                sensor_freq = self.multi_temporal_sensors['rlbench'].get(cam_name)
                assert sensor_freq is not None and sensor_freq > 0, f"FPS for camera not found: {cam_name}"
                self.last_img_obs_dict[cam_name] = deque(maxlen=1)
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.env.action_shape)
        self.tanh_action_cfg = tanh_action_cfg
        if tanh_action_cfg is not None and tanh_action_cfg.use:
            self.tanh_action_low = np.array(tanh_action_cfg['low'])
            self.tanh_action_high = np.array(tanh_action_cfg['high'])

        self.proprio_key = proprio_key

        self.multi_task = True
        assert task_descriptions is not None, 'Multi-task envs require some task description.'
        self.task_name = task_name
        self.task_variation = task_variation
        self.task_descriptions = task_descriptions
        self.multi_task_obs_dict = {
            'task': self.task_name + f'_var_{self.task_variation}',
            'task_descriptions': self.task_descriptions,
        }

        self.action_size = 4
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_size,))
        self._steps = 0

    @property
    def env_id(self):
        return f'{self.task_name}_var_{self.task_variation}'

    @property
    def spec(self):
        return EnvSpec(1, self.action_size, self.horizon)

    def observation(self, observation):
        # This function creates observations based on the current state of the environment.
        # Argument `observation` is ignored, but `gym.ObservationWrapper` requires it.
        obs_dict = self._extract_obs(observation)
        return obs_dict

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        # state = obs.get_low_dim_data()
        gripper_open = obs.gripper_open
        proprio = np.r_[obs.gripper_pose[:3], [gripper_open, 1 + gripper_open]]

        if self.use_low_state_as_proprio:
            proprio = np.copy(obs.get_low_dim_data())

        obs_dict = {
            'proprio': proprio,
            self.proprio_key: proprio,
        }
        for camera_name in self.camera_names:
            if camera_name[-4:] != '_rgb':
                camera_name = f'{camera_name}_rgb'

            if self.multi_temporal_sensors is not None and self.multi_temporal_sensors['use']:
                if self._steps == 0:
                    img = getattr(obs, camera_name)
                    self.last_img_obs_dict[camera_name].clear()
                    self.last_img_obs_dict[camera_name].append(img)
                    new_img = True
                elif self._steps % self.multi_temporal_sensors['rlbench'][camera_name] == 0:
                    img = getattr(obs, camera_name)
                    self.last_img_obs_dict[camera_name].clear()
                    self.last_img_obs_dict[camera_name].append(img)
                    new_img = True
                else:
                    assert len(self.last_img_obs_dict[camera_name]) == 1
                    img = self.last_img_obs_dict[camera_name][0]
                    new_img = False
            else:
                img = getattr(obs, camera_name)
            obs_dict[camera_name] = img

        if self.multi_task:
            obs_dict.update(self.multi_task_obs_dict)

        return obs_dict

    def render(self, mode='human') -> Union[None, np.ndarray]:
        if mode != self._render_mode:
            raise ValueError(
                'The render mode must match the render mode selected in the '
                'constructor. \nI.e. if you want "human" render mode, then '
                'create the env by calling: '
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                'You passed in mode %s, but expected %s.' % (
                    mode, self._render_mode))
        if mode == 'rgb_array':
            frame = self._gym_cam.capture_rgb()
            frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
            return frame

    def reset(self) -> Dict[str, np.ndarray]:
        if self.task._task.name.startswith('insert_onto_square_peg'):
            descriptions, obs = self.task.reset()
        elif self.task._task.name.startswith('place_shape_in_shape_sorter'):
            did_place = False
            num_tries, max_tries = 0, 1000
            while not did_place and num_tries < max_tries:
                try:
                    descriptions, obs = self.task.reset()
                    did_place = True
                except (WaypointError, TaskEnvironmentError) as e:
                    num_tries += 1
                    print(f'Could not place shape sorting objects. Num tries: {num_tries} shape_sort')
                    print(e)

        elif self.task._task.name.startswith('take_usb_out_of_computer_mrest'):
            descriptions, obs = self.task.reset()
        elif self.task._task.name.startswith('pick_and_lift_small'):
            descriptions, obs = self.task.reset()
        else:
            raise ValueError(f'Invalid task name: {self.task._task.name}')

        self._descriptions = descriptions  # Not used.
        self._steps = 0
        self._did_open_forcefully = False
        self._did_open_forcefully_step = -1
        return self._extract_obs(obs)
    
    def unnormalize_action(self, action):
        action = (action + 1) / 2.0
        # Only scale position
        if len(self.tanh_action_high) == 3:
            action[:3] = action[:3] * (self.tanh_action_high - self.tanh_action_low) + self.tanh_action_low
        else:
            action = action * (self.tanh_action_high - self.tanh_action_low) + self.tanh_action_low
        return action

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:

        if self.tanh_action_cfg is not None and self.tanh_action_cfg.use:
            action = self.unnormalize_action(action)

        # Clip actions to not go beylow certain z value
        ee_pose = self.env._robot.arm.get_tip().get_pose()
        if (ee_pose[2] + action[2]) <= (self.env._scene._workspace_minz + 0.004):
            # 2 mm above
            action[2] = self.env._scene._workspace_minz + 0.002 - ee_pose[2]
            # Grasp
            action[3] = 0.0
            # print(f"Updated action: {np.array_str(action, precision=3, suppress_small=True)}")
        
        if self.task._task.name == 'insert_onto_square_peg_mrest':
            waypoints = self.task._scene.task.get_waypoints()
            final_waypoint_pos = waypoints[-1]._waypoint.get_position()
            square_ring = [s for s in self.env._scene.pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE)
                        if s.get_name() == 'square_ring'][0]
            square_ring_pos = square_ring.get_position()

            x_error_th = 0.004
            y_error_th = 0.004
            # print(f"Final waypoint pos: {np.array_str(final_waypoint_pos[:2], precision=4, suppress_small=True)}")
            # print(f"   Square ring pos: {np.array_str(square_ring_pos[:2], precision=4, suppress_small=True)}")
            if (np.abs(square_ring_pos[0] - final_waypoint_pos[0]) < x_error_th and
                np.abs(square_ring_pos[1] - final_waypoint_pos[1]) < y_error_th):
                true_action_delta = final_waypoint_pos - square_ring_pos
                action[:2] = true_action_delta[:2]
                action[3] = 1.0
                print("**** Open gripper forcefully ****")
                self._did_open_forcefully = True
                if self._did_open_forcefully_step == -1:
                    self._did_open_forcefully_step = self._steps
            else:
                self._did_open_forcefully = False
                self._did_open_forcefully_step = -1

        elif self.task._task.name == 'place_shape_in_shape_sorter_mrest':
            # waypoints = self.task._scene.task.get_waypoints()
            # final_waypoint_pos = waypoints[-1]._waypoint.get_position()
            if self.task._scene.task.variation_index == 0:
                final_waypoint_pos = np.copy(self.env._scene.task.drop_points[0].get_pose()[:3])
            elif self.task._scene.task.variation_index == 1:
                final_waypoint_pos = np.copy(self.env._scene.task.drop_points[1].get_pose()[:3])
            else:
                raise ValueError('Invalid variation')

            target_shape = self.task._scene.task.shapes[self.task._scene.task.variation_index]
            target_pos = target_shape.get_position()

            x_error_th, y_error_th = 0.005, 0.005
            # x_error_th, y_error_th = 0.01, 0.01
            debug = False
            if (np.abs(target_pos[0] - final_waypoint_pos[0]) <= x_error_th and
                np.abs(target_pos[1] - final_waypoint_pos[1]) <= y_error_th):

                # true_action_delta = final_waypoint_pos - target_pos
                true_action_delta = final_waypoint_pos - ee_pose[:3]
                action[:2] = true_action_delta[:2]

                open_gripper_error_th_x, open_gripper_error_th_y = 0.003, 0.003
                # open_gripper_error_th_x, open_gripper_error_th_y = 0.005, 0.005
                if debug:
                    print("<<<< NEAR: Move to target >>>>") 
                    print(f"ee_pos: {np.array_str(ee_pose[:3], precision=3, suppress_small=True)} "
                        f"target_pos: {np.array_str(target_pos[:2], precision=3, suppress_small=True)}, "
                        f"waypoint_pos: {np.array_str(final_waypoint_pos[:2], precision=3, suppress_small=True)}")
                if (np.abs(target_pos[0] - final_waypoint_pos[0]) <= open_gripper_error_th_x and
                    np.abs(target_pos[1] - final_waypoint_pos[1]) <= open_gripper_error_th_y):
                    action[3] = 1.0
                    print("**** Open gripper forcefully ****")
                    self._did_open_forcefully = True
                    if self._did_open_forcefully_step == -1:
                        self._did_open_forcefully_step = self._steps
                else:
                    self._did_open_forcefully = False
                    self._did_open_forcefully_step = -1

            else:
                if debug:
                    print(f"ee_pos: {np.array_str(ee_pose[:3], precision=3, suppress_small=True)} "
                        f"Far, target_pos: {np.array_str(target_pos[:2], precision=3, suppress_small=True)}, "
                        f"waypoint_pos: {np.array_str(final_waypoint_pos[:2], precision=3, suppress_small=True)}")
        
        elif self.task._task.name == 'take_usb_out_of_computer_mrest_usb_inhand':
            if ee_pose[2] + action[2] < 1.01:
                action[2] = 1.01 - ee_pose[2]
                action[3] = 0.0

        obs, reward, terminate, _ = self.task.step(action)
        success, _ = self.task._task.success()
        # Below code is only for visualization 
        visualize = False
        if visualize:
            if self.task._task.name == 'insert_onto_square_peg_mrest' and self._did_open_forcefully_step > 5:
                # Let the object be inserted
                if success and self._steps - self._did_open_forcefully_step < 5:
                    success = False
                    terminate = False
            if self.task._task.name == 'place_shape_in_shape_sorter_mrest' and self._did_open_forcefully_step > 10:
                if success and self._steps - self._did_open_forcefully_step < 5:
                    success = False
                    terminate = False

        info = {'success': success }
        # print(f"Step: {self._steps}, true succ: {self.task._task.success()[0]} our succ: {success}")
        self._steps += 1
        return self._extract_obs(obs), reward, terminate, info

    def close(self) -> None:
        self.env.shutdown()
    


def create_rlbench_env_with_name(job_data: Mapping[str, Any], task_info, camera_names,
                                 headless: bool = True):
    task_name = task_info.env
    variation = task_info.variation

    img_size = job_data.env_kwargs.image_width
    assert job_data.env_kwargs.image_width == job_data.env_kwargs.image_height

    record_state = False

    def _camera_config(camera_name: str):
        rgb_camera_config = CameraConfig(rgb=True, depth=False, point_cloud=False,
                                        image_size=(img_size, img_size), mask=False)
        null_camera_config = CameraConfig()
        null_camera_config.set_all(False)
        if camera_names[0][-4:] == '_rgb':
            return rgb_camera_config if f'{camera_name}_rgb' in camera_names else null_camera_config
        else:
            return rgb_camera_config if camera_name in camera_names else null_camera_config

    obs_config = ObservationConfig(
        left_shoulder_camera=_camera_config('left_shoulder'),
        right_shoulder_camera=_camera_config('right_shoulder'),
        overhead_camera=_camera_config('overhead'),
        wrist_camera=_camera_config('wrist'),
        front_camera=_camera_config('front'),
        state=record_state,)
    obs_config.set_all_low_dim(True)

    renderer = 'opengl3'
    if renderer == 'opengl':
        obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
        obs_config.overhead_camera.render_mode = RenderMode.OPENGL
        obs_config.wrist_camera.render_mode = RenderMode.OPENGL
        obs_config.front_camera.render_mode = RenderMode.OPENGL
    
    task_files = [t.replace('.py', '') for t in os.listdir(rlbench_task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    assert task_name in task_files, f'Task {task_name} not recognised!.'

    task = task_file_to_task_class(task_name)

    action_absolute_mode = False
    if job_data['rlbench_envs']['data_subsampling_cfg'].get('only_predict_waypoints') is not None:
        action_absolute_mode = job_data['rlbench_envs'].data_subsampling_cfg.only_predict_waypoints.use
    add_one_hot_timestep_config = None
    if job_data['rlbench_envs']['data_subsampling_cfg'].get('add_one_hot_timestep') is not None:
        add_one_hot_timestep_config = job_data['rlbench_envs'].data_subsampling_cfg.add_one_hot_timestep
    

    env = Environment(
        # action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        action_mode=MoveArmPosOnlyThenGripper(
            EndEffectorPoseViaPlanning(absolute_mode=action_absolute_mode, only_pos=True),
            # EndEffectorPoseViaIK(absolute_mode=action_absolute_mode, only_pos=True),
            Discrete()),
        obs_config=obs_config,
        headless=headless)
    if task_name == 'pick_and_lift_small_mrest':
        env.launch('task_design_pick_and_lift_small.ttt')
    elif task_name == 'take_usb_out_of_computer_mrest':
        env.launch('task_design.ttt')
    else:
        env.launch('task_design_small_square_insert.ttt')
    
    task = task_file_to_task_class(task_name)
    task = env.get_task(task)
    task.set_variation(variation)

    if task_name == 'place_shape_in_shape_sorter_mrest':
        did_place = False
        num_tries, max_tries = 0, 1000
        while not did_place and num_tries < max_tries:
            try:
                descriptions, obs = task.reset()
                did_place = True
            except (WaypointError, TaskEnvironmentError) as e:
                num_tries += 1
                print(f'Could not place shape sorting objects. Num tries: {num_tries}, {task_name}')
    else:
        descriptions, obs = task.reset()

    use_low_state_as_proprio = False
    if job_data['rlbench_envs'].get('data_subsampling_cfg', None) is not None:
        use_low_state_as_proprio = job_data['rlbench_envs']['data_subsampling_cfg'].get('add_low_dim_state', False)

    task_env = RLBenchDictObs(
        env, task, img_size, img_size, camera_names, 
        proprio_size=4,
        task_name=task_name,
        task_variation=variation,
        task_descriptions=descriptions,
        proprio_key='ee_xyz',
        use_low_state_as_proprio=use_low_state_as_proprio,
        add_one_hot_timestep_config=add_one_hot_timestep_config,
        tanh_action_cfg=job_data['env_kwargs'].get('tanh_action'),
        multi_temporal_sensors=job_data.get('multi_temporal_sensors', None),)

    return task_env 
