from collections.abc import Callable, Iterable, Mapping
import numpy as np
import copy
import pickle
import time
from cv_bridge import CvBridge
from pathlib import Path
from typing import Any, Optional
from pathlib import Path
from collections import deque

from frankapy.demos.camera_utils import *
from frankapy.demos.camera import Camera
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage, CartesianImpedanceSensorMessage
from franka_interface_msgs.msg import SensorDataGroup, RobotState
from sensor_msgs.msg import JointState

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from threading import RLock, Thread, Lock


desired_delta_position_mutex = Lock()
desired_delta_position = np.array([0, 0, 0])


class RealtimeController(Thread):
    def __init__(self,
                 franka_arm,
                 group: None = None,
                 target = None,
                 name = None, args = ...,
                 kwargs = None, *, daemon: bool = None) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.fa = franka_arm
        self.workspace_walls = FC.WORKSPACE_WALLS[:, :3]
    
    def run(self):
        print("Running controller loop")
        time_threshold_in_seconds = 300.0
        init_time = rospy.Time.now().to_time()
        pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)

        i = 0
        total_desired_delta = np.zeros((3,))
        curr_pose = copy.deepcopy(self.fa.get_pose())
        self.fa.goto_pose(curr_pose, duration=time_threshold_in_seconds + 2.0, dynamic=True, buffer_time=10)
        start_pos_str = f"start pos: {curr_pose.translation[0]}, {curr_pose.translation[1]}, {curr_pose.translation[2]}"

        while True:
            if i % 50 == 0:
                print("In while loop")

            curr_pose = copy.deepcopy(self.fa.get_pose())
            timestamp = rospy.Time.now().to_time() - init_time
            if timestamp > time_threshold_in_seconds:
                print(f"Time exceeded threshold: {time_threshold_in_seconds}")
                break

            desired_delta_position_mutex.acquire(blocking=True)
            desired_pos = np.copy(desired_delta_position)
            desired_delta_position_mutex.release()

            # total_desired_delta += desired_pos[:3]
            next_pos = curr_pose.translation + desired_pos[:3]
            # next_pos = curr_pose.translation + total_desired_delta

            if i % 50 == 0:
                print(start_pos_str)
                print(f'controller pos: {curr_pose.translation[0]}, {curr_pose.translation[1]}, {curr_pose.translation[2]}')
                rospy.loginfo(f'controller desired_delta: {desired_pos[0]}, {desired_pos[1]}, {desired_pos[2]}')
                print(f'controller desired_delta: {desired_pos[0]:.6f}, {desired_pos[1]:.6f}, {desired_pos[2]:.6f}')
            # next_pos_clipped = np.clip(next_pos, 
            #                            self.workspace_walls.min(axis=0),
            #                            self.workspace_walls.max(axis=0),)
            next_pos_clipped = next_pos

            # desired_delta_position = np.array([0, 0, 0])
            # print(f"Desired delta: {np.array_str(desired_delta_position, precision=4, suppress_small=True)}")

            traj_gen_proto_msg = PosePositionSensorMessage(
                id=i, timestamp=timestamp, 
                position=next_pos_clipped, 
                quaternion=curr_pose.quaternion,)
            ros_msg = make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            )
            # rospy.loginfo('Publishing: ID {}, mouse_goal: {}'.format(
            #     traj_gen_proto_msg.id, input_device.goal_delta_x, input_device.goal_delta_y))
            pub.publish(ros_msg)
            i += 1

            # Sleep for a while
            time.sleep(0.04)


class RealWorldFrankaEnv:
    def __init__(self, task_name: str) -> None:
        # Create camera
        path_to_perception = Path('/home/mohit/projects/mohit-frankapy/frankapy/scripts/demos_pegboard/')
        # AZURE_KINECT_INTRINSICS = Path('calib/azure_kinect.intr')
        # AZURE_KINECT_EXTRINSICS = Path('calib/azure_kinect_overhead/azure_kinect_overhead_to_world.tf')
        REALSENSE_EE_TF = Path('calib/realsense_ee.tf')
        REALSENSE_INTR = Path('calib/realsense_intrinsics.intr')

        self.task_name = task_name

        self.hand_camera = Camera(
            path_to_perception / REALSENSE_INTR,
            path_to_perception / REALSENSE_EE_TF,
            is_realsense=True
        )

        print("Starting robot")
        self.fa = FrankaArm()

        self.start_joints = np.array([-0.0098496,-0.775723,0.0188426,-2.63923,-0.00064948,1.82679,0.78281])
        self.assume_obj_grasped = True

        # Fix the pegboard location for now.
        self.user_keypoint_pegboard = {
            'keypoint_T_world': np.array([0.61989582, 0.1329542,  0.17840856]),
        }
        self.orientation_before_insertion = np.array([
            [0.99999, 0.000368651, -0.000302574],
            [0.000368566, -0.99999, -0.000283239],
            [-0.000302675, 0.000283125, -1],
        ])

        self.subscriber_list = []
        self.ts = None
        self.step_idx = 0

        # Lock for observations
        self.obs_lock = RLock()
        self.obs_dict = {}
        self.realtime_controller = None

        # These are workspace limits for the EE-position only
        self.workspace_min = np.array([0.20, -0.46, -0.02])
        self.workspace_max = np.array([0.62,  0.46,  0.55])


    def reset_for_green_block_insert(self):
        if self.assume_obj_grasped:
            curr_pose = self.fa.get_pose()
            # Too low move vertically above first to avoid any collisions
            if curr_pose.translation[2] < 0.25:
                move_up = RigidTransform(
                    translation=np.array([0, 0, -0.10]), 
                    from_frame='franka_tool',
                    to_frame='franka_tool')
                self.fa.goto_pose_delta(move_up, duration=4.0)

        self.fa.reset_pose()
        self.fa.reset_joints()
        if not self.assume_obj_grasped:
            self.fa.open_gripper()
        self.reset_pose = self.fa.get_pose()

        version = 3
        assert version in (2, 3), f"Invalid version: {version}"

        if version == 2:
            pegboard_top = self.user_keypoint_pegboard['keypoint_T_world']
            # Now go to an intermediate location above pegboard (and reset orientation)
            start_tf = self.fa.get_pose()
            # Original training noise
            # start_noise_x = np.random.uniform(-0.10, -0.06)
            # start_noise_y = np.random.uniform(-0.06,  0.06)
            # Use smaller variance for eval
            start_noise_x = np.random.uniform(-0.08, -0.06)
            start_noise_y = np.random.uniform(-0.03,  0.03)
            start_tf.translation = [
                pegboard_top[0] + start_noise_x,
                pegboard_top[1] + start_noise_y,
                pegboard_top[2] + 0.24 + np.random.uniform(-0.02, 0.02),  # for long fingers
            ]
            if self.orientation_before_insertion is not None:
                start_tf.rotation = self.orientation_before_insertion
            self.fa.goto_pose(start_tf, duration=4.0)
        elif version == 3:
            pegboard_top = self.user_keypoint_pegboard['keypoint_T_world']
            start_tf = self.fa.get_pose()
            start_noise_x = -0.08
            start_noise_y = -0.04
            start_noise_z = 0.11
            start_tf.translation = [
                pegboard_top[0] + start_noise_x + np.random.uniform(-0.02, -0.02),
                pegboard_top[1] + start_noise_y + np.random.uniform(-0.00, 0.00),
                pegboard_top[2] + start_noise_z + np.random.uniform(-0.00, 0.00),  # for long fingers
            ]
            if self.orientation_before_insertion is not None:
                start_tf.rotation = self.orientation_before_insertion
            self.fa.goto_pose(start_tf, duration=4.0)
        
    def reset_for_lift_block(self):
        curr_pose = self.fa.get_pose()
        # Too low move vertically above first to avoid any collisions
        if curr_pose.translation[2] < 0.25:
            move_up = RigidTransform(
                translation=np.array([0, 0, -0.10]), 
                from_frame='franka_tool',
                to_frame='franka_tool')
            self.fa.goto_pose_delta(move_up, duration=4.0)

        self.fa.reset_joints()
        self.fa.open_gripper()
        start_pose = copy.deepcopy(self.fa.get_pose())

        reset_using_delta_pose = True
        if reset_using_delta_pose:
            # delta_x = np.random.uniform(0.04, 0.010) [for data collection]
            delta_x = np.random.uniform(0.06, 0.08)
            # delta_y = np.random.uniform(-0.04, 0.04) [for data collection]
            delta_y = np.random.uniform(-0.02, 0.02)
            # delta_z = np.random.uniform(0.24, 0.28)  [for data collection]
            delta_z = np.random.uniform(0.26, 0.26)
            move_down = RigidTransform(
                translation=np.array([delta_x, delta_y, delta_z]), 
                from_frame='franka_tool',
                to_frame='franka_tool')
            self.fa.goto_pose_delta(move_down)
        else:
            start_joints_list = [
                np.array([-0.158403, -0.302888, 0.168442, -2.60556, -0.0083354,  2.27985, 0.765479]),
                np.array([-0.158522, -0.36747,  0.16953,  -2.58772, -0.00922066, 2.1913,  0.763753]),
                np.array([-0.272671, -0.202176, 0.202521, -2.53137, -0.0476445,  2.34881, 0.671923]),
                np.array([-0.27191, -0.150011, 0.342683, -2.3797, -0.014785, 2.24681, 0.764354]),
                np.array([-0.0008, 0.0286867, 0.0723831, -2.48311, -0.00769809, 2.50955, 0.901647]),
            ]
            start_joints = start_joints_list[np.random.choice(len(start_joints_list))].copy()
            # Do not use any noise here (used during training)
            # start_joints[:5] += np.random.uniform(-0.01, 0.01, size=5)
            self.fa.goto_joints(start_joints)

            reset_pose_with_start_orientation = copy.deepcopy(self.fa.get_pose())
            reset_pose_with_start_orientation.rotation = start_pose.rotation
            self.fa.goto_pose(reset_pose_with_start_orientation)


    def reset(self):
        # Assume that the block is grasped
        print("Reset")
        if 'peg_insert' in self.task_name:
            self.reset_for_green_block_insert()
        elif self.task_name in ('lift_blue_block', 'lift_green_block', 'lift_yellow_block'):
            self.reset_for_lift_block()
        else:
            raise ValueError(f"Invalid task name: {self.task_name}")

        # Clear steps and any observation stuff
        self.step_idx = 0
        self.gripper_open = True
        self.obs_dict = {}


    def start_record_data(self):
        hand_image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        static_image_sub = message_filters.Subscriber('/rgb/image_raw', Image)
        robot_state = message_filters.Subscriber('/robot_state_publisher_node_1/robot_state', RobotState)
        gripper_state = message_filters.Subscriber('/franka_gripper_1/joint_states', JointState)

        queue_size = 10
        slop = 0.5

        # Create path to save demo data
        self.subscriber_list = [hand_image_sub, static_image_sub, robot_state, gripper_state]
        self.ts = message_filters.ApproximateTimeSynchronizer(
            self.subscriber_list,
            queue_size=queue_size,
            slop=slop,
            allow_headerless=False)
        self.ts.registerCallback(self.data_callback)
        rospy.loginfo("Did start recording data")
    
    def stop_record_data(self):
        for sub in self.subscriber_list:
            sub.unregister()
        self.ts = None
    
    def data_callback(self, hand_img, static_img, robot_state, gripper_state):
        # rospy.loginfo("Did receive data")
        # print("Did receive data")

        br = CvBridge()
        hand_image = br.imgmsg_to_cv2(hand_img)
        static_image = br.imgmsg_to_cv2(static_img)

        # gripper_close = abs(gripper_state.position[1] - gripper_state.position[0]) < 0.036
        gripper_open = self.gripper_open

        current_robot_state_data = {
            'O_T_EE': np.array(robot_state.O_T_EE),
            'q': np.array(robot_state.q),
            'dq': np.array(robot_state.dq),
            'O_F_ext_hat_K': np.array(robot_state.O_F_ext_hat_K),
            'K_F_ext_hat_K': np.array(robot_state.K_F_ext_hat_K),
            'gripper_width': np.array([robot_state.gripper_width]),
            'gripper_max_width': np.array([robot_state.gripper_max_width]),
            'gripper_is_grasped': np.array([robot_state.gripper_is_grasped]),

            'gripper_joints': np.array([gripper_state.position[0], gripper_state.position[1]]),
            # 'gripper_open': np.array([1 - int(gripper_close)]),
            'gripper_open': np.array([gripper_open], dtype=np.int32),

            'step_idx': self.step_idx,
        }

        obs_dict = {
            'hand': hand_image,
            'static': static_image,
            'robot_state': current_robot_state_data,
        }
        self.obs_lock.acquire()
        self.obs_dict = obs_dict
        self.obs_lock.release()
        return obs_dict

    def get_observation(self):
        """Get the current observation."""
        self.obs_lock.acquire()
        obs_dict = copy.deepcopy(self.obs_dict)
        self.obs_lock.release()
        return obs_dict
    
    def clip_action_within_virtual_walls(self, action, final_position):
        final_pos = np.clip(final_position, 
                            FC.WORKSPACE_WALLS[:, :3].min(axis=0),
                            FC.WORKSPACE_WALLS[:, :3].max(axis=0),
                            )
        return final_pos
    
    def do_one_step_motion(self, action):
        curr_pose = copy.deepcopy(self.fa.get_pose())
        final_pos = curr_pose.position + action[:3]
        clipped_final_pos = self.clip_action_within_virtual_walls(action, final_pos)
        curr_pose.translation = clipped_final_pos
        # Inference speed is also 0.05
        self.fa.goto_pose(curr_pose, duration=0.1)
    
    def start_motion_threaded(self):
        self.realtime_controller = RealtimeController(self.fa)
        self.realtime_controller.start()
    
    def do_continuous_motion(self, action):
        global desired_delta_position_mutex
        global desired_delta_position
        desired_delta_position_mutex.acquire(blocking=True)
        desired_delta_position = np.copy(action)
        desired_delta_position_mutex.release()

    def start_dynamic_motion_with_policy_cb(self, policy_cb):
        print("**** Will start motion ****")
        # time_threshold_in_seconds = 40.0
        time_threshold_in_seconds = 120.0
        init_time = rospy.Time.now().to_time()
        pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)

        start_pose = copy.deepcopy(self.fa.get_pose())
        
        K_trans = [400.0, 400.0, 400.0]
        K_rot = [30.0, 30.0, 30.0]
        self.fa.goto_pose(start_pose, duration=time_threshold_in_seconds + 2.0, dynamic=True, buffer_time=10,
                          cartesian_impedances=K_trans + K_rot)
        start_pos_str = f"start pos: {start_pose.translation[0]}, {start_pose.translation[1]}, {start_pose.translation[2]}"
        self.gripper_open = True

        i = 0
        last_time = time.time()
        action_queue = deque(maxlen=3)
        while True:
            curr_pose = self.fa.get_pose()
            timestamp = rospy.Time.now().to_time() - init_time
            if timestamp > time_threshold_in_seconds:
                print(f"Time exceeded threshold: {time_threshold_in_seconds}")
                break

            desired_delta_position = policy_cb() 
            action_queue.append(desired_delta_position)
            total_delta = np.zeros_like(desired_delta_position)
            gripper_action = desired_delta_position[-1]
            for action_hist_idx, action_in_queue in enumerate(action_queue):
                # TODO: Should we only do this initially?
                # if i > 100 and action_hist_idx < 2:
                #     continue
                total_delta[:3] += action_in_queue[:3]
            # total_delta = total_delta / len(action_queue)

            # desired_pos = curr_pose.translation + desired_delta_position[:3]
            # NOTE: This is too fragile (with 3cm it barely moved sometimes)
            total_delta[:3] = np.clip(total_delta[:3], -0.06, 0.06)
            desired_pos = curr_pose.translation + total_delta[:3]

            # Clip the desired position to be within the virtual walls
            desired_pos = np.clip(desired_pos, self.workspace_min, self.workspace_max)

            print(start_pos_str)
            print(f"\t        non-norm action (curr) \t: {np.array_str(desired_delta_position, precision=6, suppress_small=True)}, gripper: {gripper_action:.1f}")
            print(f"\t non-norm action (total delta) \t: {np.array_str(total_delta, precision=6, suppress_small=True)}, gripper: {gripper_action:.1f}")
            print(f'\t               controller pos  \t: {np.array_str(curr_pose.translation, precision=6, suppress_small=True)}')
            print(f'\t       controller desired pos  \t: {np.array_str(desired_pos, precision=6, suppress_small=True)}')

            traj_gen_proto_msg = PosePositionSensorMessage(
                id=i, timestamp=timestamp, 
                position=desired_pos,
                # quaternion=curr_pose.quaternion
                quaternion=start_pose.quaternion
            )
            ros_msg = make_sensor_group_msg(
            trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
            )
            # rospy.loginfo('Publishing: ID {}, mouse_goal: {}'.format(
            #     traj_gen_proto_msg.id, input_device.goal_delta_x, input_device.goal_delta_y))
            pub.publish(ros_msg)
            i += 1
            curr_time = time.time()
            # How long does a single loop for dynamic motion take?
            print(f"Dynamic motion one loop time: {curr_time - last_time:.6f}")
            last_time = curr_time

            if gripper_action < 0.1 and self.gripper_open:
                self.fa.close_gripper(block=False)
                self.gripper_close = True
            
        term_proto_msg = ShouldTerminateSensorMessage(
            timestamp=rospy.Time.now().to_time() - init_time, should_terminate=True)
        ros_msg = make_sensor_group_msg(termination_handler_sensor_msg=sensor_proto2ros_msg(
            term_proto_msg, SensorDataMessageType.SHOULD_TERMINATE))
        pub.publish(ros_msg)

        self.stop_record_data()
