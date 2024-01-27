#!/bin/bash

# Install the appropriate conda env and activate it
# conda create -n py37 python=3.7
# conda activate py37

PROJECT_ROOT='/home/mohit/projects/multi-resolution-real-time-control'

python -m pip install gym==0.13.0
python -m pip install mujoco==2.3.5
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
python -m pip install timm==0.6.12

# Install procedural metaworld for coarse tasks
mkdir -p $PROJECT_ROOT
cd $PROJECT_ROOT
git clone --recursive git@github.com:iamlab-cmu/mrest-rlbench.git
cd $PROJECT_ROOT/metaworld
python -m pip install -e .

# Install Robosuite which we use for programmatically creating scenes
git clone --recursive git@github.com:SaumyaSaxena/robosuite.git $PROJECT_ROOT/robosuite
cd $PROJECT_ROOT/robosuite
git checkout sawyer_multitask
python -m pip install -r requirements.txt
python -m pip install -e .


# Install PyRep
git clone https://github.com/stepjam/PyRep.git $PROJECT_ROOT/pyrep
cd $PROJECT_ROOT/pyrep
python -m pip install -r requirements.txt
python -m pip install -e .

# Install RLBench (for precision tasks)
git clone --recursive git@github.com:iamlab-cmu/mrest-rlbench.git $PROJECT_ROOT/rlbench
cd $PROJECT_ROOT/rlbench
python -m pip install -r requirements.txt
python -m pip install -e .


# Install PyBullet and our dynamic ballbot environment for dynamic manipulation
git clone git@github.com:iamlab-cmu/mrest-pybullet.git $PROJECT_ROOT/mrest-pybullet
cd mrest-pybullet
python -m pip install -e .

# NOTE: Install mujoco_py
# For some old dependencies of metaworld you may have to install mujoco_py
INSTALL_MUJOCOPY=0
if [[ "$INSTALL_MUJOCOPY" == "1" ]]; then
    git clone --recursive git@github.com:openai/mujoco-py.git $PROJECT_ROOT/mujoco-py
    cd $PROJECT_ROOT/mujoco-py
    # NOTE: This can fail sometimes and you may have to run extra commands.
    python setup.py install
fi

# python -m pip install transformers
cd $PROJECT_ROOT/mrest/
# Assume the directory is correctly installed
python -m pip install -e .

