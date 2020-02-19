# Consensus optimization

This repository includes the code used to generate numerical results in the ICASSP 2020 conference paper titled "Decentralized optimization with non-identical sampling in presence of stragglers".

*The snapshot of the code/data/plots can be found in this [release](https://github.com/thadikari/consensus/releases/tag/v2.0).*


| MNIST | Fashion-MNIST |
|:-------------------------:|:-------------------------:|
| <img width="100%" src="data/archive10_icassp_final_results/run_mnist_linear1_distinct_PWG_perfect_amb_iclr_10_bern_08_60_10_metro.png?raw=true">  |   <img width="100%" src="data/archive10_icassp_final_results/run_fashion_mnist_linear1_distinct_PWG_perfect_amb_iclr_10_bern_08_60_10_metro.png?raw=true">|
|<img width="100%"  src="data/archive10_icassp_final_results/run_mnist_linear1_distinct_PWG_rand_walk_amb_iclr_10_bern_08_60_10_metro.png?raw=true">  |  <img width="100%"  src="data/archive10_icassp_final_results/run_fashion_mnist_linear1_distinct_PWG_rand_walk_amb_iclr_10_bern_08_60_10_metro.png?raw=true">|
|<img width="100%"  src="data/archive10_icassp_final_results/run_mnist_relu1_distinct_PWG_rand_walk_amb_iclr_10_bern_08_60_10_metro.png?raw=true">  |  <img width="100%"  src="data/archive10_icassp_final_results/run_fashion_mnist_relu1_distinct_PWG_perfect_amb_iclr_10_bern_08_60_10_metro.png?raw=true">|

## Recreating results in paper
#### Generating data:
* Execute [`run_main.sh`](run_main.sh) to run all simulations included in the paper.
* Sample usage for running simulations individually:
```
python -u run_main.py --dataset fashion_mnist --strategy distinct --func linear1 --opt PWG --consensus perfect --strag_dist bern --strag_dist_param 0.8 --num_samples 60 --grad_combine Equal Proportional --save --graph_def amb_iclr_10 --num_iters 5000
python -u run_main.py --dataset cifar10 --strategy distinct --func conv --opt PWG --consensus perfect --strag_dist bern --strag_dist_param 0.8 --num_samples 60 --grad_combine Equal Proportional --save --graph_def amb_iclr_10 --num_iters 5000 --max_loss_eval_size 2 --lrate_start 0.001 --lrate_end 0.0001 --weights_scale 0.08
```
* Results are saved in `~/SCRATCH`.

#### Generating plots:
Use the following to generate all plots.
* `python plot_run_main.py --ylog --num_iters 100000 --no_dots --silent --save --keywords linear1 perfect --xhide --fig_size 6.5 2.08 --ylim 0.25 100`
* `python plot_run_main.py --ylog --num_iters 100000 --no_dots --silent --save --keywords linear1 rand_walk --all_workers --xhide --fig_size 6.5 2.08 --ylim 0.35 100 --filter_sigma 5`
* `python plot_run_main.py --ylog --num_iters 100000 --no_dots --silent --save --keywords relu1 perfect --fig_size 6.5 2.52 --ylim 0.2 1`



-----



## Comparing the two upper bounds
The paper suggests a sufficient condition for the proportional weighting to converge faster than equal weighting, by comparing the upper bounds the variances of "equal" and "proportional" weighting. The condition includes three terms. The following plots compare how the terms including $\mu_2$ and $n^2 \mu_3$ vary for different toy distributions. Note that the latter is smaller than the former as suggested in the paper. Since $D$ is not measurable a scalar (0.01) is used in the following.


| Terms vs. straggler distribution parameter | Histogram (~PDF) of straggler distribution |
|:-------------------------:|:-------------------------:|
| <img width="100%" src="data/archive9_theoretically_compare_equal_vs_proportional_upper_bnds/bern__60_1.png?raw=true">  |   <img width="100%" src="data/archive9_theoretically_compare_equal_vs_proportional_upper_bnds/bern__60_1_hist_0.8.png?raw=true">|
| <img width="100%" src="data/archive9_theoretically_compare_equal_vs_proportional_upper_bnds/exptime__1200_15.png?raw=true">  |   <img width="100%" src="data/archive9_theoretically_compare_equal_vs_proportional_upper_bnds/exptime__1200_15_hist_10.png?raw=true">|
| <img width="100%" src="data/archive9_theoretically_compare_equal_vs_proportional_upper_bnds/gauss__200_400.png?raw=true">  |   <img width="100%" src="data/archive9_theoretically_compare_equal_vs_proportional_upper_bnds/gauss__200_400_hist_30.png?raw=true">|

* Sample commands for regenerating plots:
```
python plot_dist_mu2.py bern --trials 1000000 --bern_max 200 --bern_min 50
python plot_dist_mu2.py gauss --trials 1000000 --gauss_loc 200 --gauss_max_std 400
python plot_dist_mu2.py exp --trials 1000000 --exp_max 60 --exp_max_scale 15
python plot_dist_mu2.py exptime --trials 1000000 --exptime_max_scale 15 --exptime_b0 1200 --exptime_t0 20
```
* Change content in [`plot_dist_mu2.sh`](plot_dist_mu2.sh) and execute to generate all plots.
* All plots are located at [`data/archive9_theoretically_compare_equal_vs_proportional_upper_bnds`](data/archive9_theoretically_compare_equal_vs_proportional_upper_bnds).

## Variance of the gradients
This section is about computing the variance of gradient, $\sigma^2$, for a toy example.
* Run `python run_main.py --dataset toy_model --opt PG --consensus perfect --grad_combine Equal Proportional --save --weights_seed 1 --eval_grad_var --num_iters 1 --strag_dist bern --strag_dist_param 0.8 --num_samples 60 --func linear3 --strategy toy_4_3 --toy_sigma2 10.0`.
* The `toy_4_3` model assumes 4 compute nodes, 3 dimensional vectors.
* The parameter `toy_sigma2` adjusts the variance within a distribution.
* See [`model_toy.py`](src/model_toy.py) for details.
* Generate data for multiple `toy_sigma2` values by with [`run_variance.sh`](run_variance.sh).
* Plot the results using `python plot_run_main.py --xlog --ylog --type plot_var --name toy_lylx`.



-----



## Other related experiments - Graph vs. distribution
* `python run_main.py --dataset mnist --strategy PQQQ --graph_def wk_4 --consensus rand_walk --num_consensus_rounds 5 --strag_dist equal --num_samples 64 --grad_combine Equal --weights_seed 99 --save --loss_eval_freq 10`

