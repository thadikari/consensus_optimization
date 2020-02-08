# Consensus optimization

This repository includes the code used to generate numerical results in the ICASSP 2020 conference paper titled "Decentralized optimization with non-identical sampling in presence of stragglers".

| MNIST | Fashion-MNIST |
|:-------------------------:|:-------------------------:|
| <img width="100%" src="data/archive10_icassp_final_results/run_mnist_linear1_distinct_10_PWG_perfect_amb_iclr_10_bern_08_60_10_metro.png?raw=true">  |   <img width="100%" src="data/archive10_icassp_final_results/run_fashion_mnist_linear1_distinct_10_PWG_perfect_amb_iclr_10_bern_08_60_10_metro.png?raw=true">|
|<img width="100%"  src="data/archive10_icassp_final_results/run_mnist_linear1_distinct_10_PWG_rand_walk_amb_iclr_10_bern_08_60_10_metro.png?raw=true">  |  <img width="100%"  src="data/archive10_icassp_final_results/run_fashion_mnist_linear1_distinct_10_PWG_rand_walk_amb_iclr_10_bern_08_60_10_metro.png?raw=true">|
|<img width="100%"  src="data/archive10_icassp_final_results/run_mnist_relu1_distinct_10_PWG_rand_walk_amb_iclr_10_bern_08_60_10_metro.png?raw=true">  |  <img width="100%"  src="data/archive10_icassp_final_results/run_fashion_mnist_relu1_distinct_10_PWG_perfect_amb_iclr_10_bern_08_60_10_metro.png?raw=true">|

## Recreating results in paper
#### Generating data:
* Sample usage: `python -u run_main.py --model fashion_mnist --data_dist distinct_10 --func linear1 --opt PWG --consensus perfect --strag_dist bern --strag_dist_param 0.8 --num_samples 60 --grad_combine Equal Proportional --save --graph_def amb_iclr_10 --num_iters 5000`
* `python -u run_main.py --model cifar10 --data_dist distinct_10 --func conv --opt PWG --consensus perfect --strag_dist bern --strag_dist_param 0.8 --num_samples 60 --grad_combine Equal Proportional --save --graph_def amb_iclr_10 --num_iters 5000 --max_loss_eval_size 2 --lrate_start 0.001 --lrate_end 0.0001 --weights_scale 0.08`
* Execute [`run_main.sh`](run_main.sh) to run all simulations included in the paper.

#### Generating plots:
Use the following to generate all plots.
* `python plot_run_main.py --ylog --num_iters 100000 --no_dots --silent --save --keywords linear1 perfect --xhide --fig_size 6.5 2.08 --ylim 0.25 100`
* `python plot_run_main.py --ylog --num_iters 100000 --no_dots --silent --save --keywords linear1 rand_walk --all_workers --xhide --fig_size 6.5 2.08 --ylim 0.35 100 --filter_sigma 5`
* `python plot_run_main.py --ylog --num_iters 100000 --no_dots --silent --save --keywords relu1 perfect --fig_size 6.5 2.52 --ylim 0.2 1`




-----



## Other experiments

#### Useful commands:

* `python run_main.py --model mnist --consensus rand_walk --strag_dist round --save --loss_eval_freq 10 --num_consensus_rounds 10 --doubly_stoch metro`
* `python run_main.py --model mnist --consensus rand_walk --strag_dist round --loss_eval_freq 10 --graph_def wk_4 --numw 4`

### Graph vs. distribution
* `python run_main.py --model mnist --data_dist PQQQ --graph_def wk_4 --consensus rand_walk --num_consensus_rounds 5 --strag_dist equal --num_samples 64 --grad_combine Equal --weights_seed 99 --save --loss_eval_freq 10`


### Compare variances of "equal weighting" and "proportional weighting"
#### Comparing theoretical upper bounds
* Sample commands:
```
python plot_dist_mu2.py bern --trials 1000000 --bern_max 200 --bern_min 50
python plot_dist_mu2.py gauss --trials 1000000 --gauss_loc 200 --gauss_max_std 400
python plot_dist_mu2.py exp --trials 1000000 --exp_max 60 --exp_max_scale 15
```
* Change content in [`plot_dist_mu2.sh`](plot_dist_mu2.sh) and execute to generate all plots.
* See results at [`data/archive9_theoretically_compare_equal_vs_proportional_upper_bnds`](data/archive9_theoretically_compare_equal_vs_proportional_upper_bnds).

#### Computing variance of the gradients given by two methods
* Run `python run_main.py --model toy --opt PG --consensus perfect --grad_combine Equal Proportional --save --weights_seed 1 --eval_grad_var --num_iters 1 --strag_dist bern --strag_dist_param 0.8 --num_samples 60 --func linear3 --data_dist distinct_4_3 --toy_sigma2 10.0`.
* The `distinct_4_3` model assumes 4 compute nodes, 3 dimensional vectors.
* The parametere `toy_sigma2` adjusts the variance within a distribution.
* See [`model_toy.py`](src/model_toy.py) for details.
* Generate data for multiple `toy_sigma2` values by with [`run_variance.sh`](run_variance.sh).
* Plot the results using `python plot_run_main.py  --xlog --ylog --type plot_var --name toy_lylx`.