#!/bin/bash

export OMP_NUM_THREADS=2

torchrun --nproc_per_node=1 --nnodes=2 --node_rank=$RANK --rdzv_id=456 --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT pretrain.py --launch_type=ddp --checkpoint_iters=400
# --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT pretrain.py --launch_type=ddp --start_from_checkpoint
