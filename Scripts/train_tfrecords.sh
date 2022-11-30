#!/bin/bash 
# Bash script to run CNN-LSTM for 3D sleep classification project

export CUDA_VISIBLE_DEVICES=8
echo "$CUDA_VISIBLE_DEVICES"

# main arguments
mode=test
model=model_dau_nores
dataset=dataloader_sleep_tfrecords_local
seed=42

# dataloader arguments
data_path=/shared/planck/MRI/xiaohui/mouse_optical/sleep_stage/2022-Ben-10s-raw-masked-tempzscore-broadband-states-tfrecords
batch_size=4
#if [[ $mode == "gradcam" || $mode == "test" || $mode == "test_subjectwise" ]]; then
if [ $mode != "train" ]; then
    batch_size=1
fi
echo "$batch_size"

num_classes=3
num_frames=168

# network arguments
num_epochs=200
lr_init=0.0001
num_rnn_units=64
loss=focal
#model_name=cnnlstm_seed_${seed}_bs_8_class_${num_classes}_epochs_${num_epochs}_lr_${lr_init}_frames_${num_frames}_loss_${loss}_nores_nosim_onemorelayers
model_name=cnnlstm_seed_${seed}_bs_8_class_${num_classes}_epochs_${num_epochs}_lr_${lr_init}_frames_${num_frames}_loss_${loss}_nores_nosim_onemorelayers_ben_pool_all_exclude2918_rawmask_tempzscore_lstmatt_rnn64_hemis
model_savedir=../Models/${model_name}
logs_dir=../Logs/${model_name}

# results saving arguments
result_savedir=../Results/result_outputs/
result_action=save
gradcam_label=1

echo "$mode"
echo "$model_savedir"

python main.py --mode $mode --model $model --dataset $dataset --seed $seed --data_path $data_path --batch_size $batch_size --num_epochs $num_epochs --lr_init $lr_init --model_savedir $model_savedir --result_savedir $result_savedir --result_action $result_action --num_classes $num_classes --num_rnn_units $num_rnn_units --gradcam_label $gradcam_label --logs_dir $logs_dir --loss $loss --num_frames $num_frames --hemispheric
