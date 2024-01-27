import mujoco
import os
import hydra
from omegaconf import DictConfig, OmegaConf


if 'USE_RLBENCH' not in os.environ:
    os.environ['USE_RLBENCH'] = '0'
print(f"==== Should be using RLBench: {os.environ['USE_RLBENCH']} ====")

if 'USE_PYBULLET' not in os.environ:
    os.environ['USE_PYBULLET'] = '0'
print(f"==== Should be using Pybullet: {os.environ['USE_PYBULLET']} ====")


from run_eval_on_train_ckpts import bc_eval_loop_on_train_checkpoints
from train_loop_with_encoder_multi_task import bc_train_multitask
from test_loop_multitask_generalization import bc_eval_with_finetune_multitask

cwd = os.getcwd()

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(version_base="1.1.0", config_name="BC_train_multitask_config", config_path="config")
def configure_jobs(job_data:dict) -> None:
    print("in job")
    os.environ['GPUS'] = os.environ.get('SLURM_STEP_GPUS', '0')
    
    print("========================================")
    print("Job Configuration")
    print("========================================")

    job_data = OmegaConf.structured(OmegaConf.to_yaml(job_data))
    job_data['cwd'] = cwd
    with open('job_config.json', 'w') as fp:
        OmegaConf.save(config=job_data, f=fp.name)

    if job_data['mode'] == 'train':
        bc_train_multitask(job_data)
    elif job_data['mode'] == 'eval':
        bc_eval_loop_on_train_checkpoints(job_data)
    elif job_data['mode'] == 'test_finetune':
        bc_eval_with_finetune_multitask(job_data)
    else:
        print("Requested mode not available currently")
        raise NotImplementedError


if __name__ == "__main__":
    # Spawned processes don't work well with wandb (cause live loading)
    # multiprocessing.set_start_method('spawn')
    configure_jobs()
