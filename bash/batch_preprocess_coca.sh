#!/bin/bash                                                                 
#SBATCH -c 6
#SBATCH -t 72:00:00
#SBATCH --mem=10G
#SBATCH -p evlab
#SBATCH -n 1
#SBATCH --array=1-200%200
#SBATCH -o logs/slurm.%A-%a.out # STDOUT

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
    python -m composlang --path "$file" --tag "${tag}.db" --cache_dir cache/skip_plus_triplets_batched_lowercase --batch_size 5_000
fi
