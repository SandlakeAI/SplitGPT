#!/bin/bash

# Launch script for dual-node training on Lambda Labs
# Usage: 
#   On MASTER node: ./launch_dual_node.sh master <WORKER_NODE_IP>
#   On WORKER node: ./launch_dual_node.sh worker <MASTER_NODE_IP>

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 {master|worker} <other_node_ip>"
    echo "  On master node: $0 master <worker_ip>"
    echo "  On worker node: $0 worker <master_ip>"
    exit 1
fi

ROLE=$1
OTHER_NODE_IP=$2
MASTER_PORT=29500

# Get this machine's IP (usually the first non-loopback IP)
THIS_NODE_IP=$(hostname -I | awk '{print $1}')

if [ "$ROLE" = "master" ]; then
    MASTER_ADDR=$THIS_NODE_IP
    NODE_RANK=0
    echo "Starting MASTER node at $THIS_NODE_IP, worker at $OTHER_NODE_IP"
elif [ "$ROLE" = "worker" ]; then
    MASTER_ADDR=$OTHER_NODE_IP
    NODE_RANK=1
    echo "Starting WORKER node at $THIS_NODE_IP, master at $OTHER_NODE_IP"
else
    echo "Role must be 'master' or 'worker'"
    exit 1
fi

echo "Master address: $MASTER_ADDR:$MASTER_PORT"
echo "Node rank: $NODE_RANK"

# Launch training
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=splitgpt_training \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --node_rank=$NODE_RANK \
    train_gpt.py

echo "Training completed on $ROLE node" 