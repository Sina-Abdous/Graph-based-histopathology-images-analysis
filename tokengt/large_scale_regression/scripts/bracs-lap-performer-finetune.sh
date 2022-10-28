#!/usr/bin/env bash

ulimit -c unlimited

# Change dataset-name for each local run

fairseq-train \
--user-dir ../tokengt \
--num-workers 16 \
--ddp-backend=legacy_ddp \
--dataset-name /home/re/Documents/Dr.RohbanLab/cell_graphs/ \  
--dataset-source dgl-local \
--task graph_prediction \
--criterion l1_loss \
--arch tokengt_base \
--lap-node-id \
--lap-node-id-k 16 \
--lap-node-id-sign-flip \
--performer \
--performer-finetune \
--performer-feature-redraw-interval 100 \
--prenorm \
--num-classes 1 \
--attention-dropout 0.0 --act-dropout 0.1 --dropout 0.0 \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.1 \
--lr-scheduler cosine --warmup-updates 1000 --max-update 200000 \
--lr 2e-4 \
--batch-size 128 \
--data-buffer-size 20 \
--save-dir ./ckpts/bracs-tokengt-lap16-performer-finetune \
--tensorboard-logdir ./tb/bracs-tokengt-lap16-performer-finetune \
--no-epoch-checkpoints
