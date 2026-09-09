#!/usr/bin/env bash
#SBATCH --job-name=haloscope-official
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/official-%j.out
#SBATCH --error=logs/official-%j.err

set -euo pipefail

STAGE="${1:-generate}"
OFFICIAL_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
CURRENT_ROOT="${HALOSCOPE_REIMPLEMENTATION_ROOT:-$OFFICIAL_ROOT/../LLM_Haloscope}"
PYTHON="${HALOSCOPE_PYTHON:-$CURRENT_ROOT/.venv/bin/python}"

cd "$OFFICIAL_ROOT"

export PYTHONPATH="$OFFICIAL_ROOT/.deps:$OFFICIAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$CURRENT_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p logs save_for_eval models

LLAMA_CACHE="$HF_HUB_CACHE/models--meta-llama--Llama-2-7b-chat-hf"
LLAMA_PATH=$(find "$LLAMA_CACHE/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)

if [[ -z "$LLAMA_PATH" || ! -f "$LLAMA_PATH/config.json" ]]; then
    echo "ERROR: Llama snapshot was not found under $LLAMA_CACHE"
    exit 2
fi
if [[ ! -f "$OFFICIAL_ROOT/models/BLEURT-20/config.json" ]]; then
    echo "ERROR: models/BLEURT-20 is missing or incorrect"
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python executable was not found: $PYTHON"
    exit 2
fi

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Stage: $STAGE"
echo "Node: $(hostname)"
echo "Python: $PYTHON"
echo "Llama: $LLAMA_PATH"
nvidia-smi

COMMON_ARGS=(
    --dataset_name tqa
    --model_name llama2_chat_7B
    --model_dir "$LLAMA_PATH"
    --most_likely 1
)

case "$STAGE" in
    generate)
        srun "$PYTHON" hal_det_llama.py "${COMMON_ARGS[@]}" --num_gene 1 --gene 1
        ;;
    label)
        srun "$PYTHON" hal_det_llama.py "${COMMON_ARGS[@]}" --use_rouge 0 --generate_gt 1
        ;;
    detect)
        srun "$PYTHON" hal_det_llama.py \
            "${COMMON_ARGS[@]}" --use_rouge 0 --weighted_svd 1 --feat_loc_svd 3
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Use: generate, label, or detect"
        exit 2
        ;;
esac
