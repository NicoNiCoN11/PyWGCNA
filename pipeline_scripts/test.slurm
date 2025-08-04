#!/bin/bash
#SBATCH --job-name=WGCNA
#SBATCH --output=logs/WGCNAnegesr_%j.out
#SBATCH --error=logs/WGCNAnegesr_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate wgcna
python /home/jiguo/PyWGCNA/pipeline_scripts/pyWGCNA.py