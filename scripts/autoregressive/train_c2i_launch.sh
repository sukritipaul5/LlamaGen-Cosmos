#!/usr/bin/env bash
#SBATCH --tasks=2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --job-name=tokenise-imnet-7
#SBATCH --output=logs/llamagencosmos-%j.out
#SBATCH --wait-all-nodes=1
#SBATCH --exclusive



NODES_ARRAY=($(scontrol show hostnames $SLURM_JOB_NODELIST))
HEAD_NODE=${NODES_ARRAY[0]}
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | awk '{print $1}')

echo HEAD_NODE_IP = $HEAD_NODE_IP

# envs
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1
export NCCL_CROSS_NIC=1
export NCCL_IB_DISABLE=1 # Nexus does not have Infiniband
export NCCL_SOCKET_IFNAME=bond0
export PYTHONIOENCODING=UTF-8

# debug
export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

echo $pwd
NGPUS=$(nvidia-smi -L | wc -l)
if [ $NGPUS -eq 0 ]; then
    echo "No GPU found"
    exit 1
fi

#export dirs
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_TOKEN="your-huggingface-token-here"  # Replace with your HF token
export TMPDIR="${TMPDIR:-/tmp/llamagen}"
mkdir -p $TMPDIR

echo "SLURM_PROCID = $SLURM_PROCID"
WORLD_SIZE=$SLURM_JOB_NUM_NODES
echo "WORLD_SIZE = $WORLD_SIZE"
#HF CLI Login
huggingface-cli login --token $HF_TOKEN

set -x

export WANDB_API_KEY="your-wandb-api-key-here"  # Replace with your Weights & Biases API key

export LAUNCH="
    torchrun 
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv-endpoint $HEAD_NODE_IP:12384 \
    --nnode $WORLD_SIZE \
    --nproc_per_node 4 \
    "

export SCRIPT="./autoregressive/train/train_c2i.py"
export SCRIPT_ARGS="--cosmos_path ./data/cosmos-imagenet-shards \
--vocab-size 64000 \
--dataset imagenet_code \
--cloud-save-path ./checkpoints \
--no-compile \
--global-batch-size 192 --ckpt-every 10000 --epochs 300 \
--no-local-save \
--gpt-ckpt ./checkpoints/model-checkpoints/ \
--wandb-id YOUR_WANDB_ID \
--wandb-name YOUR_RUN_NAME"

export PYTHONPATH=".:$PYTHONPATH"

# launch job
export CMD="$LAUNCH $SCRIPT $SCRIPT_ARGS $@"
srun $CMD

