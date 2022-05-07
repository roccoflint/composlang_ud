#!/bin/bash                                                                 
#SBATCH -c 6
#SBATCH -t 4:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu
#SBATCH -p evlab
#SBATCH -n 1
#SBATCH --array=1-500%100
#SBATCH -o logs/batch_LM_score/slurm.%A-%a.out # STDOUT

source activate-conda
conda activate composlang

set -x
cd ~/code/composlang/notebooks
python score_csv.py -i ./pairs_with_ctxt.csv -m gpt2 -bsz 64 --part "$SLURM_ARRAY_TASK_ID" --total_parts 500
