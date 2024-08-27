#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=tom_ppo1
#SBATCH --time=10:00:00
#SBATCH --account=def-foutsekh
#SBATCH --output=tom_ppo1.txt

module load anaconda3
source activate good_generalist



python monobeast.py --monobeast 1 --stilltrain 1 --finetune 2_f --eval 0 --train_data_path /scratch/f/foutsekh/nstevia/bug_localization/bug_rep_paulina/train_with_metrics_Tomcat_dataset_baseline.csv --project_name Tomcat  --algo IMPPPO



python monobeast.py --monobeast 1 --finetune 2_f --eval 1 --train_data_path /scratch/f/foutsekh/nstevia/bug_localization/bug_rep_paulina/test_with_metrics_Tomcat_dataset_baseline.csv --project_name Tomcat  --algo IMPPPO
