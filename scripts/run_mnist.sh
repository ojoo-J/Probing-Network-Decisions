#!/bin/bash

# python -m experiments.eval_clf \
#     --dataset MNIST \
#     --data-dir /project/prev/NNV/data \
#     --save-dir /project/outputs \
#     --arch CNN \
#     --ckpt-path /project/run/MNIST_CNN.pth \
#     --seed 0 \
#     --layer-name fc_layer.4 \
#     --acc-n 1

# python -m experiments.train_prober \
#     --dataset MNIST \
#     --train-path /project/outputs/MNIST_fc_layer.4_acc1_train_60000.pkl \
#     --valid-path /project/outputs/MNIST_fc_layer.4_acc1_valid_10000.pkl \
#     --save-dir /project/outputs/smoothing-0.2_lr-1e-2 \
#     --epochs 30 \
#     --batch-size 128 \
#     --lr 1e-2 \
#     --label-smoothing 0.2 \
#     --latent-dims 256 128 64 \
#     --split mirror

python -m experiments.generate_counterfactual \
    --data-dir /project/prev/NNV/data \
    --dataset MNIST \
    --save-dir /project/run/outputs/valid/false_miss \
    --steps 1000 \
    --batch-size 128 \
    --lr 1e-4 \
    --seed 0 \
    --device cuda \
    --cls-ckpt-path /project/run/MNIST_CNN.pth \
    --prober-ckpt-path /project/outputs/smoothing-0.2_lr-1e-2/prober_ep-16_lr-0.01_acc-0.9787_fpr-0.1733.pth \
    --prober-dims 256 128 64 \
    --prober-split mirror \
    --prober-train-path /project/outputs/MNIST_fc_layer.4_acc1_train_60000.pkl \
    --prober-valid-path /project/outputs/MNIST_fc_layer.4_acc1_valid_10000.pkl \
    --prober-layer-name fc_layer.4 \
    --g-ckpt-path /project/run/MNIST_realNVP.pth \
    --index-path /project/outputs/smoothing-0.2_lr-1e-2/split.json &