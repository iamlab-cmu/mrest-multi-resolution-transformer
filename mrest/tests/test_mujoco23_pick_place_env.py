import numpy as np
from moviepy.editor import ImageSequenceClip

try:
    import cv2
except:
    print('OpenCV not found')

from mrest.envs.sawyer_pick_place_multitask import SawyerPickAndPlaceMultiTaskGenEnvV2


if __name__ == "__main__":
    
    env_config = {
        "target_object": "blockB", 
        "stack_on_object": "blockA", 
        "skill": "push_left", 
        "task": "push_left",
        "only_use_block_objects": True}

    env = SawyerPickAndPlaceMultiTaskGenEnvV2(env_config, data_collection=True)
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    use_opencv_to_render = False
    save_gif = True

    n_trajs = 1
    total_succ = 0
    H, W = 256, 256
    cam_name = 'left_cap2'
    for i in range(n_trajs):
    
        env = SawyerPickAndPlaceMultiTaskGenEnvV2(env_config, data_collection=True)
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        step = 0
        
        env.reset_model()
        obs = env.reset()
        img = env.render(offscreen=True, camera_name=cam_name, resolution=(H, W))
        # img = img[::-1, :, :]
        success = False
        imgs = [img]
        
        while (not success) and (step < 200):
        # while (step < 200):
            a = np.random.uniform(low=0.0, high=10.0, size=4)
            a = np.array([1,0,0,0])
            # a = policy.get_action(obs)
            # a = np.random.normal(a, 0.07 * 2)
            next_obs, r, done, info = env.step(a)
            # img = env.get_image()
            img = env.render(offscreen=True, camera_name=cam_name, resolution=(H, W))
            # img = img[::-1, :, :]

            if use_opencv_to_render:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow(f'{type(env).__name__}', img)
                cv2.waitKey(1)

            imgs.append(img)
            obs = next_obs.copy()
            step += 1
            if info['success']:
                success = info['success']
                success = True
                print(f"================SUCCESS at step: {step}===============")
                total_succ += 1

        if save_gif:
            print("saving gif")
            skill=env_config['skill']
            tg=env_config['target_object']
            filename = f'./media/sawyer_{skill}_gen_{i}_{success}_{tg}.gif'
            cl = ImageSequenceClip(imgs, fps=20)
            cl.write_gif(filename, fps=20)
        
    print(f"Total success: {total_succ}/{n_trajs}")
    if use_opencv_to_render:
        cv2.destroyAllWindows()
