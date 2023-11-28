#!/bin/bash
for i in {1..3}; do
    python main.py --n_rounds 150 --num_clients 100 --local_ep 1 --lr 1e-4 --n_sampled 3 --n_data_train 500 --n_data_val 100 --bandit --q 0.1 --iid covariate1 --dataset fashion-mnist
    python main.py --n_rounds 150 --num_clients 100 --local_ep 1 --lr 1e-4 --n_sampled 3 --n_data_train 500 --n_data_val 100 --bandit_var --q 0.5 --q_min 0.07 --iid covariate1 --dataset fashion-mnist
    python main.py --n_rounds 150 --num_clients 100 --local_ep 1 --lr 1e-4 --n_sampled 3 --n_data_train 500 --n_data_val 100 --DAC --tau 30 --iid covariate1 --dataset fashion-mnist
    python main.py --n_rounds 150 --num_clients 100 --local_ep 1 --lr 1e-4 --n_sampled 3 --n_data_train 500 --n_data_val 100 --random --iid covariate1 --dataset fashion-mnist
    python main.py --n_rounds 150 --num_clients 100 --local_ep 1 --lr 1e-4 --n_sampled 3 --n_data_train 500 --n_data_val 100 --oracle --iid covariate1 --dataset fashion-mnist
done