# Multi-Resolution Sensing for Real-Time Control with Vision-Language Models

The repository contains the training and evaluation code from the CoRL 2023 paper [Multi-Resolution Sensing for Real-Time Control with Vision-Language Models](https://mohitsharma0690.github.io/multi-res-real-time-control/).

## Installation

Create a conda environment with `python >= 3.7`.

```
conda create -n py37 python=3.7
conda activate py37
```

Install the main code as an editable python package.

```
git clone --recursive git@github.com:iamlab-cmu/mrest-multi-resolution-transformer.git mrest
cd mrest
python -m pip install -e .
```

For installing all the other dependencies (including simulation environments) please look at `install.sh`. For installing each environment dependency please look at environment installation section below. 


## Environment Installations

We use three different environments to evaluate trained models. Detailed instructions on installing each of these environments is provided in [install.sh]().

**MT-Coarse**: For coarse tasks we focus on MetaWorld environments. However, we update default metaworld code to use the latest mujoco version which allows us to use Google Scanned Objects in our environment. To use coarse environments please install our fork of [MetaWorld from here](https://github.com/iamlab-cmu/mrest-metaworld).


**MT-Precise**: For precise tasks we use 4 different set of tasks from RLBench. To use these tasks we first need to download CoppeliaSim and install [PyRep](https://github.com/stepjam/PyRep). Once PyRep is installed, please install our [fork of RLBench](https://github.com/iamlab-cmu/mrest-rlbench). Our fork of RLBench builds on top of [HiveFormer's fork of RLBench](https://github.com/guhur/RLBench/tree/74427e188cf4984fe63a9c0650747a7f07434337).  

**MT-Dynamic**: For dynamic tasks we use our custom PyBullet based ballbot environment. Please find installation and usage instructions at our [pybullet branch repo](https://github.com/iamlab-cmu/mrest-pybullet).


## Downloading Demonstration Data

All data for each simulation task and real-world task is located here: https://drive.google.com/drive/folders/1O3ggQrhlAv5GLackUux_iC0xurP1T9W5?usp=drive_link. This folder contains separate folders to each environment type (see details below). Each environment type has a tar.gz file that you need to download and untar. The location of this untarred file then needs to be specified in its respective environment yaml file, i.e., `core/config/metaworld_envs`, or `core/config/pybullet_envs`.

**MT-Coarse**: https://drive.google.com/drive/folders/1U5hSpncEW2XWxmKoudvqThzqsu9ayIxQ?usp=drive_link

**MT-Precise**: https://drive.google.com/drive/folders/1ZqdBJjU77yMx4BWyKaDMXjcqGOKBw8Oi?usp=drive_link

**MT-Dynamic**: https://drive.google.com/drive/folders/1CO0cGBD3FEyK6q34dCX9C4TUOYx7fFD6?usp=drive_link


## Run Train Code

To train on coarse metaworld tasks please run:

```
python ../core/hydra_launcher.py --config-name=BC_train_multitask_config epochs=60 \
  agent_eval.use=False wandb.saver_no_eval.use=True \
  env_kwargs.tanh_action.use=False embedding=mdetr_multiview \
  env_type=metaworld bc_kwargs.loss_type=MSE \
  image_encoder_kwargs.mdetr_multiview.image_augmentations.eye_in_hand_90.train.color_jitter=True \
  image_encoder_kwargs.mdetr_multiview.image_augmentations.eye_in_hand_90.train.stochastic_jitter=True
```

To train on dynamic ballbot tasks please run:

```
python ../core/hydra_launcher.py --config-name=BC_train_multitask_config epochs=60 \
  agent_eval.use=False wandb.saver_no_eval.use=True \
  env_kwargs.tanh_action.use=False embedding=mdetr_multiview \
  env_type=pybullet bc_kwargs.loss_type=MSE \
  image_encoder_kwargs.mdetr_multiview.image_augmentations.eye_in_hand_90.train.color_jitter=True \
  image_encoder_kwargs.mdetr_multiview.image_augmentations.eye_in_hand_90.train.stochastic_jitter=True
```

## Run Eval Code

We can run evaluation code while the model is training, however, since we have many environments to evaluate overall training can be slow. To speed up we recommend running evaluation code asynchronously while training happens. 

For this please use the hydra_launcher script with `BC_eval_on_train_ckpts_config` config.We can specify the checkpoint `checkpoint.run_path` and how many trajectories to run for each task `eval_num_traj.train` among other things. 

Finally, note that for fast evaluation we can evaluate multiple checkpoints in parallel. 
- **Number of eval runs**: To achieve this you first need to set the total number of eval runs that should happen simultaneously. For this set `run_epoch.total=4`. What this parameter does is it will divide all the checkpoints into `run_epoch.total` sets.
- **Current eval run**: Then you need to specify which set of checkpoints to evaluate. For this set `run_epoch.current=0` to evaluate the first set of checkpoints.

As training and evaluation happen simultaneously and asynchronously, the eval script waits for a fixed duration of time `sleep_time` before checking if there are any new checkpoints to evaluate. This happens repeatedly until a total cutoff time.

The overall command to run evaluation is then as follows:

```
python ../../core/hydra_launcher.py --config-name=BC_eval_on_train_ckpts_config \
  seed=4 run_epoch.current=0 gpu_id=0 \
  checkpoint.run_path=iam-lab/visual-repr-manip/2gbjivgs \
  env_gif_saver.save_env_freq=10 mode=eval run_epoch.use=True \
  run_epoch.total=2 sleep_time=300 eval_num_traj.train=5 eval_num_traj.heldout=0 
```

We have also provided a convenience script in `./mrest/bash/eval_train_ckpts/run_multiple_evals.sh`. Please check the script for more details.


## Citation

If you use this code in your research, please consider citing our paper:

```
@inproceedings{saxena2023multi,
  title={Multi-Resolution Sensing for Real-Time Control with Vision-Language Models},
  author={Saxena, Saumya and Sharma, Mohit and Kroemer, Oliver},
  booktitle={Conference on Robot Learning},
  pages={2210--2228},
  year={2023},
  organization={PMLR}
}
```

### Thanks

Our code builds on top of many impressive works. We would like to thank the authors of [MDETR](https://github.com/ashkamath/mdetr), [CLIP](https://openai.com/research/clip), [R3M](https://github.com/facebookresearch/r3m), for making their code available. We also extensively use multiple manipulation simulators including MuJoCo, PyBullet and CoppeliaSim. We would like to thank the many developers who have contributed to these simulators.
