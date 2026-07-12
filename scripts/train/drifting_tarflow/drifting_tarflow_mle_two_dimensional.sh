#!/bin/bash

# Slurm setup
#SBATCH -p gu-compute
#SBATCH -A gu-account
#SBATCH --qos=gu-med
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=4-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

FILE_NAME="drifting_tarflow_mle_two_dimensional.py"
OUT_DIR="./logs/${FILE_NAME}"
mkdir -p "${OUT_DIR}"

DATE_WITH_TIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUT_DIR}/${DATE_WITH_TIME}_run_${SLURM_JOB_ID}.log"

exec > "$OUTPUT_FILE" 2>&1

echo "==================================="
date
echo "Job running on node: $(hostname)"
echo "==================================="

source ./venv/bin/activate
echo "[DEBUG] Python check:"
python --version

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "[DEBUG] PYTHONPATH: $PYTHONPATH"
echo "==================================="

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "${FILE_NAME} starts at $(date)"
echo "==================================="

# exprt WANDB_API_KEY and WANDB_ENTITY from .env
echo "Wandb Settings"
set -a
source .env
set +a

export WANDB_CONFIG_DIR="$(pwd)/.wandb_config"
export WANDB_DIR="$(pwd)/wandb_logs"


echo "==================================="

# Args
dataset_name=spiral
img_size=8
channel_size=1
num_flow_blocks=8
# hidden_dim=512
flow_block_dim=8
num_attn_blocks=8
permutation_type=flip
attn_num_heads=8
attn_temp=1.0
ffn_expansion=4
# cfg_weight=0.0
batch_size=1024
epochs=20000
drift_loss_weight=0.5
lr=1e-4
lr_schedule_type=ws
sample_freq=100
# num_samples=3000
resume_wandb_url=false
# annealed_guidance_flag=""



python -u train/${FILE_NAME} \
    --dataset_name "$dataset_name" \
    --img_size "$img_size" \
    --channel_size "$channel_size" \
    --num_flow_blocks "$num_flow_blocks" \
    --flow_block_dim "$flow_block_dim" \
    --num_attn_blocks "$num_attn_blocks" \
    --permutation_type "$permutation_type" \
    --attn_num_heads "$attn_num_heads" \
    --attn_temp "$attn_temp" \
    --ffn_expansion "$ffn_expansion" \
    --batch_size "$batch_size" \
    --epochs "$epochs" \
    --drift_loss_weight "$drift_loss_weight" \
    --lr "$lr" \
    --lr_schedule_type "$lr_schedule_type" \
    --sample_freq "$sample_freq" \
    --resume_wandb_url "$resume_wandb_url" 

echo "-----------------------------------"

echo "==================================="
echo "Fin at $(date)"