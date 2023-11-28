#!/bin/bash
python train_target_2B_fusion_optic.py \
    --gpus '1' \
    --num_workers 2 \
    --batch_size 2 \
    --test_batch_size 2 \
    --dataset 'G1020' \
    --data_root './save_new_data/G1020/DeepLabv3p/train_200/' \
    --epochs 100 \
    --save_vis_epoch False \
    --use_filtering True \
    --use_confident_filtering True \
    --use_balance_filtering True \
    --use_noisy_filtering False \
    --base_threshold 0.9 \
    --num_warm_up 10 \
    --filtering_case 'case3' \
    --use_balance_filtering True \
    --balance_times 5 \
    --mode_balance 'random_cluster' \
    --num_balance_cluster 5 \
    --noisy_seg_loss_function 'CE' \
    --Dice_CE_alpha 5.0 \
    --Dice_CE_beta 1.0 \
    --w_clean_seg 1.0 \
    --w_noisy_seg 1.0 \
    --exp_name 'test'