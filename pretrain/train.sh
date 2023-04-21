#!/bin/bash

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=$RANK --rdzv_id=456 --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT pretrain.py --launch_type=ddp