# RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph

This repository is the official implementation of RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph (ACL 2025)
# DBM
## Requirements
- Python 3.11.3
- pytorch 2.0.1  
- numpy 1.24.3  
- scikit-learn 1.2.2 

## Usage
All training commands are listed in best_config.sh. For example, you can run the following commands to train TransE-RSCF-diag on FB15k-237 and WN18RR datasets.

```
bash run.sh train TransE FB15k-237 0 Diag 1024 256 1000 9.0 1.0 0.00005 100000 16 --hp 11 --rp 5 --r_weight 0.1
bash run.sh train TransE wn18rr 0 Diag 512 1024 500 6.0 0.5 0.00005 80000 8 --hp 3 --rp 3 --r_weight 0.05
```
# TDM
## Requirements
- Python 3.6.13
- pytorch 1.7.0  
- numpy 1.19.5  
- tqdm 4.64.1

or 
- Python 3.11.3
- pytorch 2.0.1  
- numpy 1.24.3  
- tqdm 4.64.1

## Usage
All training commands are listed in best_config.sh. For example, you can run the following commands to train ComplEX-DURA-RSCF on FB15k-237, WN18RR and YAGO3-10 datasets.

```
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset FB237 --model ComplEx --rank 2000 --optimizer Adagrad --learning_rate 1e-1 --batch_size 100 --regularizer DURA_W --reg 5e-2 --max_epochs 200 --valid 5 -train -id 0 -save -RSCF -hp 4 -r_weight 1
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset WN18RR --model ComplEx --rank 2000 --optimizer Adagrad --learning_rate 1e-1 --batch_size 100 --regularizer DURA_W --reg 1e-1 --max_epochs 50 --valid 5 -train -id 0 -save -weight -RSCF -hp 4 -rp 10
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset YAGO3-10 --model ComplEx --rank 1000 --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer DURA_W --reg 5e-3 --max_epochs 200 --valid 5 -train -id 0 -save -RSCF -hp 3 -rp 5
```

## Acknowledgement
We refer to the code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) and [DURA](https://github.com/MIRALab-USTC/KGE-DURA) Thanks for their contributions.
