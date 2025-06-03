# Dual-Node Training Setup for Lambda Labs

This guide explains how to run the GPT training script across two 8xH100 nodes on Lambda Labs.

## Quick Setup

### 1. Rent Two Nodes
- Rent 2x 8xH100 nodes from Lambda Labs
- Make sure they're in the same region/availability zone for best network performance
- Note down the IP addresses of both nodes

### 2. Setup on Both Nodes
On each node, ensure you have:
```bash
# Your training data should be available on both nodes
# Either use shared storage or copy the data to both nodes
ls data/fineweb10B/  # Verify data exists

# Make the launch script executable
chmod +x launch_dual_node.sh
```

### 3. Launch Training

**On the MASTER node (choose one arbitrarily):**
```bash
./launch_dual_node.sh master <WORKER_NODE_IP>
```

**On the WORKER node:**
```bash
./launch_dual_node.sh worker <MASTER_NODE_IP>
```

**Example:**
- Master node IP: `10.0.1.100`  
- Worker node IP: `10.0.1.101`

On master node:
```bash
./launch_dual_node.sh master 10.0.1.101
```

On worker node:
```bash
./launch_dual_node.sh worker 10.0.1.100
```

## What Changes with Dual-Node

### Effective Batch Size
- Single node (8 GPUs): `world_size * train_seq_len = 8 * 48K = 384K tokens/batch`
- Dual node (16 GPUs): `world_size * train_seq_len = 16 * 48K = 768K tokens/batch`

The effective batch size doubles, which may affect:
- Training dynamics (larger batches can be more stable)
- Learning rate (you might want to scale it up slightly)
- Convergence speed (fewer total batches needed)

### Network Requirements
- The nodes need to communicate gradients (~768M parameters)
- Lambda Labs nodes should have high-bandwidth interconnect
- Training will be limited by the slowest network link between nodes

## Troubleshooting

### Connection Issues
```bash
# Test connectivity between nodes
ping <OTHER_NODE_IP>

# Check if the master port is accessible
nc -zv <MASTER_NODE_IP> 29500
```

### Firewall/Security Groups
Lambda Labs nodes should allow inter-node communication by default, but if you have issues:
- Ensure port 29500 is open between nodes
- Check Lambda Labs security group settings

### Data Synchronization
Make sure your training data is identical on both nodes:
```bash
# Compare data file checksums
md5sum data/fineweb10B/fineweb_train_*.bin
```

### Monitoring
- Only the master node (rank 0) will print logs and save checkpoints
- Monitor GPU utilization on both nodes: `nvidia-smi`
- Check network usage: `iftop` or `nethogs`

## Performance Expectations

With dual-node setup:
- ~2x throughput compared to single node (assuming good network)
- Some overhead from inter-node communication (typically 5-15%)
- Better memory efficiency (model replicated across more GPUs)

## Alternative: Manual torchrun

If you prefer to launch manually without the script:

**Master node:**
```bash
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=splitgpt --rdzv_backend=c10d --rdzv_endpoint=<MASTER_IP>:29500 --node_rank=0 train_gpt.py
```

**Worker node:**
```bash
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=splitgpt --rdzv_backend=c10d --rdzv_endpoint=<MASTER_IP>:29500 --node_rank=1 train_gpt.py
```

## Cost Optimization Tips

1. **Preemptible instances**: Use Lambda Labs' spot/preemptible instances if available
2. **Checkpointing**: Enable `save_checkpoint = True` in hyperparameters for fault tolerance
3. **Data locality**: Use shared storage or ensure data is pre-loaded on both nodes
4. **Monitor utilization**: Watch for any idle GPUs or network bottlenecks 