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

FILE_NAME="generate_gif.py"
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

if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "[ERROR] python 또는 python3 명령어를 찾을 수 없습니다. 작업을 종료합니다."
    exit 1
fi

echo "[DEBUG] 선택된 Python 명령어: $PYTHON_CMD"
echo "[DEBUG] Python check:"
$PYTHON_CMD --version

export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "[DEBUG] PYTHONPATH: $PYTHONPATH"
echo "==================================="

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "${FILE_NAME} starts at $(date)"
echo "==================================="

# Args
base_dir="results/train/drifting_tarflow_two_dimensional_learnable_ratio_reg/spiral"

for experiment_dir in "$base_dir"/*; do
    if [ -d "$experiment_dir" ]; then
        for final_path in "$experiment_dir"/*; do
            if [ -d "$final_path" ]; then
                echo "[INFO] Target directory: $final_path"
                $PYTHON_CMD -u plot/${FILE_NAME} \
                    --dir "$final_path" 
            fi
        done
    fi
done

echo "-----------------------------------"
echo "==================================="
echo "Fin at $(date)"