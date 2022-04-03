#!/bin/bash                                                                 
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --mem=100G
#SBATCH -p evlab
#SBATCH -n 1
#SBATCH --array=1-200
#SBATCH -o logs/slurm_output.%a.out # STDOUT

count=$(find COCA/output_of_stanza_0* -maxdepth 1 -type f|wc -l)
if [[ $SLURM_ARRAY_TASK_ID -ge $count ]]
then
    echo "$SLURM_ARRAY_TASK_ID ge $count; stopping."
    exit 0
else
    files=(COCA/output_of_stanza_0*.txt)
    file="${files[SLURM_ARRAY_TASK_ID]}"
    tag="${file##*/}"
    tag="${tag%.txt}"

    source activate-conda
    conda activate composlang

    set -x
    python -m composlang --path "$file" --tag "${tag}.db" --cache_dir cache/batched --batch_size 1000
fi