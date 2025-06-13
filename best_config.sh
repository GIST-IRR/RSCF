# Best Configuration for RotatE
bash run.sh train RotatE FB15k-237 0 Diag 1024 256 1000 9.0 1.0 0.00005 100000 16 -de --hp 11 --rp 3 --r_weight 0.1
bash run.sh train RotatE wn18rr 0 Diag 512 1024 500 6.0 0.5 0.00005 80000 8 -de --hp 5 --rp 2 --r_weight 0.01
bash run.sh train RotatE FB15k-237 0 Linear_2 1024 256 1000 9.0 1.0 0.00005 100000 16 -de --Linear_2 --hp 6 --rp 2 --r_weight 0.5
bash run.sh train RotatE wn18rr 0 Linear_2 512 1024 500 6.0 0.5 0.00005 80000 8 -de --Linear_2 --hp 7 --rp 1 --r_weight 0
# Best Configuration for TransE
bash run.sh train TransE FB15k-237 0 Diag 1024 256 1000 9.0 1.0 0.00005 100000 16 --hp 11 --rp 5 --r_weight 0.1
bash run.sh train TransE wn18rr 0 Diag 512 1024 500 6.0 0.5 0.00005 80000 8 --hp 3 --rp 3 --r_weight 0.05
bash run.sh train TransE FB15k-237 0 Linear_2 1024 256 1000 9.0 1.0 0.00005 100000 16 --Linear_2 --hp 7 --rp 4 --r_weight 0.1
bash run.sh train TransE wn18rr 0 Linear_2 512 1024 500 6.0 0.5 0.00005 80000 8 --Linear_2 --hp 13 --rp 11 --r_weight 0.1