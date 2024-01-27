import random

from PIL import Image
from pathlib import Path

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import matplotlib.pyplot as plt
import cv2


def run_metaworld():
    env_name = 'assembly-v2-goal-observable'

    # MT-10 tasks
    # env_name = 'door-close-v2-goal-observable'
    # env_name = 'drawer-open-v2-goal-observable'
    # env_name = 'pick-place-v2-goal-observable'
    # env_name = 'reach-v2-goal-observable'
    # env_name = 'window-open-v2-goal-observable'
    # env_name = 'door-open-v2-goal-observable'
    # env_name = 'peg-insert-side-v2-goal-observable'
    # env_name = 'push-v2-goal-observable'
    # env_name = 'window-close-v2-goal-observable'

    # MT-20
    # env_name = 'button-press-v2-goal-observable'
    # env_name = 'button-press-topdown-param-v2-goal-observable'
    # env_name = 'button-press-topdown-wall-v2-goal-observable'
    # env_name = 'button-press-wall-v2-goal-observable'
    # env_name = 'assembly-v2-goal-observable'
    # env_name = 'basketball-v2-goal-observable'
    # env_name = 'box_close-v2-goal-observable'
    # env_name = 'coffee-button-v2-goal-observable'
    # env_name = 'coffee-pull-v2-goal-observable'
    # env_name = 'coffee-push-v2-goal-observable'

    # env_name = 'dial-turn-v2-goal-observable'
    # env_name = 'disassemble-v2-goal-observable'

    # env_name = 'door-close-v2-goal-observable'
    # env_name = 'door-lock-v2-goal-observable'
    # env_name = 'door-open-v2-goal-observable'
    # env_name = 'door-unlock-v2-goal-observable'

    # env_name = 'hand-insert-v2-goal-observable'
    # env_name = 'faucet-open-v2-goal-observable'
    env_name = 'faucet-close-v2-goal-observable'

    seed = random.randrange(1000)
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=seed)
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    camera_name = 'left_cap2'

    # Should we flip the image? Depends on where the camera axes points.
    invert_img = True
    if camera_name in ('left_cap2', 'right_cap2', 'top_cap2'):
        invert_img = True  # for left_cap2, right_cap2, 
    elif camera_name in ('corner4_handle_press_side', ):
        invert_img = False

    use_opencv_to_render = False

    episodes_to_run = 100
    for episode_idx in range(episodes_to_run):
        env.reset()
        env.reset_model()
        total_reward = 0

        if use_opencv_to_render:
            cv2.namedWindow(f"{env_name}-{camera_name}", cv2.WINDOW_AUTOSIZE)
            # cv2.namedWindow(f"{env_name}-{camera_name}", cv2.WINDOW_NORMAL)
  
        for t in range(20):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            total_reward += reward

            img = env.sim.render(width=448, height=448, depth=False, device_id=2,
                                 camera_name=camera_name)
            # plt.imshow(img)
            # plt.show()
            if invert_img:
                img = img[::-1, :, :]

            if use_opencv_to_render:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow(f'{env_name}-{camera_name}', img)
                cv2.waitKey(1)

            else:
                img_arr = Image.fromarray(img)
                save_dir = Path(f'/home/mohit/output_test_metaworld/ep_{episode_idx:02d}')
                if not save_dir.exists():
                    save_dir.mkdir()
                img_arr.save(save_dir / f'img_{t:04d}.png')
        
        print(f"Episode finished [{episode_idx}/{episodes_to_run}], total reward: {total_reward}")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run_metaworld()
