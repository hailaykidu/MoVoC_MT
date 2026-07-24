#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --job-name="movoc_mt"
#SBATCH --output=movoc_mt_train.out
#SBATCH --error=movoc_mt_train.err
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00

source /homes/neumann/teklehaymanot/envs/tigrinya_mt/bin/activate

cd /homes/neumann/teklehaymanot/TigrinyaTokenizer/MoVoC_MT/04_training

python -u train_mt.py --output_dir ./mt_output
