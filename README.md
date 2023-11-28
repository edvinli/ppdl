# Efficient node selection in private personalized decentralized learning

To run an experiment on label shift run the following

```
python main.py --n_rounds 150 --num_clients 100 --local_ep 1 --lr 3e-5 --n_sampled 3 --n_data_train 400 --n_data_val 100 --bandit --q 0.1 --iid label --dataset cifar-10
```

--iid arguments

* label: label shift for cifar-10 (two clusters, animals and vehicles)
* label3: for both fashion-mnist and cifar-10, two random classes in each cluster (five clusters)
* covariate1: rot_deg = [0, 180, 10, 350] #cluster rotations,  cluster sizes = [0.7, 0.9, 0.95]
* covariate2: rot_deg = [0,90,180,270] #cluster rotations, cluster sizes = [0.25, 0.5, 0.75]