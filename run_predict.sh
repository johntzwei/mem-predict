#!/bin/bash
#SBATCH --job-name=probe_experiment
#SBATCH --partition=nlp_hiprio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Force single GPU to avoid peer mapping error
export CUDA_VISIBLE_DEVICES=0

# Activate your virtual environment
source /home1/lijc/mem-predict/.venv/bin/activate

# Navigate to your project directory
cd /home1/lijc/mem-predict

# Run your script
python predict/train.py --config predict/configs/gutenberg_1b.json