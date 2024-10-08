# Harnessing pre-trained generalist agents for SE tasks
Our supplementary materials to support the findings of our paper can be found in supplementary.pdf

Our predictions can be found in the predictions folder.

This repository contains the training scripts, datasets and experiments correspoding to our paper here. We forked the following repositories 
in order to set up the generalist agents:

Pytorch implementation of [IMPALA](https://github.com/facebookresearch/torchbeast.git)

Pytorch implementation of the [Multi-game Decision Transformer](https://github.com/etaoxing/multigame-dt.git)

## Requirements
Python 3.8+

## Set up
To run the experiments involving each generalist agents

`pip install requirements1.txt`

`pip install requirements1.txt`

## Pre-trained models
 Download the MGDT pre-trained model [here](gs://rl-infra-public/multi_game_dt/checkpoint_38274228.pkl)
 The IMPALA agent pre-trained model is under ./IMPALA_Pretrained/model.tar

## Run experiments 
X can be 0, 1 or 2 w.r.t the fine-tuning data budgets
### Run IMPALA experiments
 `python -m torchbeast.monobeast --finetuning X`


 ### Run MGDT experiments
 `python run.py --finetuning X`
