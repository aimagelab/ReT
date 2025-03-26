#!/bin/bash
#SBATCH --job-name=inference_m2kr_large
#SBATCH --output=
#SBATCH --error=
#SBATCH --open-mode=truncate
#SBATCH --partition=
#SBATCH --account=
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3
#SBATCH --time=00:30:00

# tested on 4 NVIDIA A100-SXM-64GB

conda activate ret
cd ~/ReT
export PYTHONPATH=.

export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True

DATASET_NAMES=(
    "okvqa"
    "infoseek"
    "evqa"
    "wit"
    "llava"
    "kvqa"
    "oven"
    "iglue"
)

JSONL_ROOT_PATH=
DATASET_PATHS=(
    "${JSONL_ROOT_PATH}/okvqa_test.jsonl"
    "${JSONL_ROOT_PATH}/infoseek_test.jsonl"
    "${JSONL_ROOT_PATH}/evqa_test_m2kr.jsonl"
    "${JSONL_ROOT_PATH}/wit_test.jsonl"
    "${JSONL_ROOT_PATH}/llava_test.jsonl"
    "${JSONL_ROOT_PATH}/kvqa_test.jsonl"
    "${JSONL_ROOT_PATH}/oven_test.jsonl"
    "${JSONL_ROOT_PATH}/iglue_test.jsonl"
)

DATASET_PASSAGES_PATHS=(
    "${JSONL_ROOT_PATH}/okvqa_passages_test.jsonl"
    "${JSONL_ROOT_PATH}/infoseek_passages_test.jsonl"
    "${JSONL_ROOT_PATH}/evqa_passages_test.jsonl"
    "${JSONL_ROOT_PATH}/wit_passages_test.jsonl"
    "${JSONL_ROOT_PATH}/llava_passages_test.jsonl"
    "${JSONL_ROOT_PATH}/kvqa_passages_test.jsonl"
    "${JSONL_ROOT_PATH}/oven_passages_test.jsonl"
    "${JSONL_ROOT_PATH}/iglue_passages_test.jsonl"
)

IMAGE_ROOT_PATH=

model_name="ReT-CLIP-ViT-L-14"
checkpoint_path="aimagelab/${model_name}"
root_path=
dataset_path="${DATASET_PATHS[$SLURM_ARRAY_TASK_ID]}"
dataset_passages_path="${DATASET_PASSAGES_PATHS[$SLURM_ARRAY_TASK_ID]}"
experiment_name="${model_name}"
index_name="${DATASET_NAMES[$SLURM_ARRAY_TASK_ID]}"

echo "DATASET PATH: ${dataset_path}"
echo "DATASET PASSAGES PATH: ${dataset_passages_path}"

srun -c $SLURM_CPUS_PER_TASK --mem $SLURM_MEM_PER_NODE \
python inference.py \
--action index \
--dataset_path $dataset_passages_path \
--image_root_path $IMAGE_ROOT_PATH \
--checkpoint_path $checkpoint_path \
--root_path $root_path \
--experiment_name $experiment_name \
--index_name $index_name \
--index_bsize 128

srun -c $SLURM_CPUS_PER_TASK --mem $SLURM_MEM_PER_NODE \
python inference.py \
--action search \
--dataset_path $dataset_path \
--dataset_passages_path $dataset_passages_path \
--image_root_path $IMAGE_ROOT_PATH \
--checkpoint_path $checkpoint_path \
--root_path $root_path \
--experiment_name $experiment_name \
--index_name $index_name \
--index_bsize 128 \
--num_docs_to_retrieve 500
