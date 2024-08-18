#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=CoT_Ensemble
#SBATCH --mem=50G
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodelist=n02
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_output/inter_SC_%j.out

source /data2/${USER}/.bashrc
source /data2/jeongseokoh/miniconda3/etc/profile.d/conda.sh
conda activate myenv
huggingface-cli login --token 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
#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
#### "llama3_8b", "gemma_7b", "mistral_7b", "qwen2_7b"
#### mmlu  race  aqua  gsm8k  math  arc  triviaqa mmlu2
#srun python idea1_SC_process_and_Gen.py --model gemma_7b --dataset race --times 20 --batchsize 2

#srun python inter_SC.py --model llama3_8b --dataset gsm8k --n_inter 10 --n_filter 5 --n_output_per_one 6

#srun python inter_SC_beam.py --model llama3_8b --dataset gsm8k --n_init 40 --n_filter 20 --n_output_per_one 2 --max_tokens 1024 --n_tokens_per_iter 150
#srun python inter_SC_beam_expand.py --model llama3_8b --dataset gsm8k --n_init 10 --n_filter 5 --n_output_per_one 2 --max_tokens 1024 --n_tokens_per_iter 50
#srun python inter_SC_beam_shrink.py --model llama3_8b --dataset gsm8k --n_init 20 --n_filter 10 --n_output_per_one 4 --max_tokens 1024 --n_tokens_per_iter 100
#srun python Original_SC.py --model llama3_8b --dataset mmlu --num_path 40

#### levenshtein , jaccard, rougeL
#### kmeans , agglomerative
srun python inter_SC_llm.py --model llama3_8b --dataset mmlu --n_init 40 --n_filter 10 --n_output_per_one 2 --max_tokens 1024 --n_tokens_per_iter 85 --distance jaccard --clustering kmeans

#srun python inter_SC_llm_linebyline.py --model llama3_8b --dataset gsm8k --n_init 10 --n_filter 5 --n_output_per_one 2 --max_tokens 1024 --n_tokens_per_iter 85 --distance levenshtein

#srun python inter_SC_llm_expand.py --model llama3_8b --dataset gsm8k --n_init 10 --n_filter 5 --n_output_per_one 2 --max_tokens 1024 --n_tokens_per_iter 85 --distance levenshtein

#srun python inter_SC_llm_to_llm.py --model llama3_8b --dataset gsm8k --n_init 40 --n_filter 5 --n_output_per_one 8 --max_tokens 1024 --n_tokens_per_iter 85 --distance levenshtein --clustering agglomerative
