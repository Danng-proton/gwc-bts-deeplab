#!/usr/bin/env bash
set -x
DATAPATH="/data/yyx/data/kitti/2015/data_scene_flow"
CUDA_VISIBLE_DEVICES=1,5 python -u main.py --dataset kitti \
    --start_epoch 149 \
    --batch_size 2 \
    --train \
    --train_deeplab \
    --sig_arg 0 \
    --datapath $DATAPATH \
    --trainlist ./filenames/kitti15_train_pseudo.txt \
    --testlist ./filenames/kitti15_train_pseudo.txt \
    --epochs 1000 --lrepochs "400,600,800:10" \
    --model gwcnet-gc \
    --mono_encoder densenet161_bts \
    --logdir ./checkpoints/kitti_have_fun \
    --mono_model_name bts_eigen_v2_pytorch_densenet161 \
    --loadckpt /data1/dyf2/gwc-deeplab-refine/GwcNet-master/checkpoints/kitti/checkpoint_000060.ckpt \
    --sig_resume /data1/dyf2/gwc-deeplab-refine/GwcNet-master/checkpoints/kitti_have_fun/sig_checkpoint_2_batch_desne_gt_000999.ckpt \
    --mono_checkpoint_path /data1/dyf2/gwc-deeplab-refine/GwcNet-master/checkpoints/mono_kitti/bts_eigen_v2_pytorch_densenet161/model \
    --test_batch_size 1 \
    --mono_max_depth 80 \
    --do_kb_crop \
    --maxdisp 192 \
    #--loadckpt ./checkpoints/kitti/only_update_bn_ft_lr4/checkpoint_000032.ckpt \
    
