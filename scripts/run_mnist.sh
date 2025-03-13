#!/bin/bash

python -m experiments.eval_clf \
    --dataset MNIST \
    --data-dir /project/prev/NNV/data \
    --save-dir /project/outputs \
    --arch CNN \
    --ckpt-path /project/run/MNIST_CNN.pth \
    --seed 0 \
    --layer-name fc_layer.4 \
    --acc-n 1

# python -m experiments.train_prober \
#     --dataset MNIST \
#     --train-path /project/outputs/MNIST_fc_layer.4_acc5_train_60000.pkl \
#     --valid-path /project/outputs/MNIST_fc_layer.4_acc5_valid_10000.pkl \
#     --save-dir /project/outputs/epochs-20_lr-1e-1 \
#     --epochs 2 \
#     --batch-size 128 \
#     --lr 1e-1 \
#     --label-smoothing 0.2 \
#     --latent-dim1 256 \
#     --latent-dim2 128 \
#     --latent-dim3 64 \
#     --split mirror

# python -m experiments.generate_counterfactual \
#     --data-dir /project/prev/NNV/data \
#     --dataset MNIST \
#     --save-dir /project/outputs \
#     --steps 5000 \
#     --batch-size 128 \
#     --lr 1e-1 \
#     --seed 0 \
#     --device cuda \
#     --cls-ckpt-path /project/run/MNIST_CNN.pth \
#     --prober-ckpt-path /project/outputs/2025-03-12_094415/prober_ep-09_lr-0.001_acc-0.9835_f1-0.9915.pth \
#     --g-ckpt-path /project/run/MNIST_realNVP.pth \
#     --index-path /project/outputs/2025-03-12_094415/split.json 