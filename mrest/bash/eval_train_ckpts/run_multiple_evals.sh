#!/bin/bash

# Specify some hydra config parameters that we want to be constant across all runs for 
# this experiment. These arguments are provided directly to each file.
args=(
    # Add: Any extra command line arguments here.
    mode=eval
    
    run_epoch.use=True
    run_epoch.total=2

    sleep_time=300
    eval_num_traj.train=2
    eval_num_traj.heldout=0

    gsheet.worksheet=multiview
    gsheet.worksheet_info=multiview_info
)

# `tasks` denotes the different task configurations we want to run, e.g. in the current code it is defined as batch sizes.
tasks=("0" "1")
# Define the CUDA devices we want to use for training.
cuda_devices=("0" "1" "0" "1" "1" "2")

seeds=("4" "4" "4" "4" "5" "5" "71" "71")

env_gif_save_freq=("10" "10" "10" "10" "10" "10" "10")

run_paths=("iam-lab/visual-repr-manip/2gbjivgs")
# Define the hydra config file we want to use for training.
config_name=BC_eval_on_train_ckpts_config


echo "${args[@]}"

i=0
conda_bash="eval $(conda shell.bash hook)"
for runpath in ${run_paths[@]}; do
for task in ${tasks[@]}; do
    echo "Running task:" $task
    echo ${cuda_devices[$i]}

    # This is the command that we will be using to run each "task" separately.
    cmd="CUDA_VISIBLE_DEVICES=${cuda_devices[$i]} python ../../core/hydra_launcher.py \
         --config-name=${config_name} \
         seed=${seeds[$i]} run_epoch.current=${task} gpu_id=${cuda_devices[$i]} checkpoint.run_path=${runpath} \
         env_gif_saver.save_env_freq=${env_gif_save_freq[$i]} \
         ${args[@]}"
    echo $cmd

    # Create a temporary executable (.sh) file that sets the appropriate environment variables,
    # sets the right conda environments, cd into the right directory and then run the cmd above.
    run_filename="tmp_${task}_${i}.sh"
    cat > ${run_filename} <<EOF
#!/bin/zsh

echo "Will run"
unset LD_PRELOAD
export COPPELIASIM_ROOT=$HOME/sims/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
eval $(conda shell.bash hook)
cd /home/mohit/projects/object_centric/mrest/mrest/bash/eval_train_ckpts
conda activate demo_tac_mujoco23
${cmd}
EOF

    # Now create a new tmux environment and run the above saved executable.
    chmod +x ${run_filename}
    sess_name="eval_${task}_${i}_${cuda_devices[$i]}"
    tmux new-session -d -s "$sess_name" /home/mohit/projects/object_centric/mrest/mrest/bash/eval_train_ckpts/${run_filename}

    i=`expr $i + 1`
    sleep 20

done
done

