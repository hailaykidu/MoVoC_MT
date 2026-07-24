#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --job-name="movoc_mt_eval"
#SBATCH --output=movoc_mt_eval.out
#SBATCH --error=movoc_mt_eval.err
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00

source /homes/neumann/teklehaymanot/envs/tigrinya_mt/bin/activate

cd /homes/neumann/teklehaymanot/TigrinyaTokenizer/MoVoC_MT/05_evaluation

python -u evaluate.py --model_dir ../04_training/mt_output/final
python -u evaluate_tigre_zeroshot.py --model_dir ../04_training/mt_output/final
