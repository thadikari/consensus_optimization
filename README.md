# Consensus optimization - ICASSP '20

## Generate data:
python main.py --data_dist distinct_10 --func linear1 --consensus perfect --strag_dist bern --strag_dist_param 0.8 --num_samples 60 --grad_combine Equal Proportional --save --graph_def amb_iclr_10 --num_iters 10000
python main.py --data_dist distinct_10 --func linear1 --consensus rand_walk --num_consensus_rounds 70 --strag_dist bern --strag_dist_param 0.8 --num_samples 60 --grad_combine Equal Proportional --save --graph_def amb_iclr_10 --num_iters 10000
python main.py --data_dist distinct_10 --func relu1 --consensus rand_walk --num_consensus_rounds 70 --strag_dist bern --strag_dist_param 0.8 --num_samples 60 --grad_combine Equal Proportional --save --graph_def amb_iclr_10 --num_iters 10000

## Generate plots:
python plot.py --ext pdf --ylog --all_workers --num_iters 4000  --no_dots

## Other:
python main.py rand_walk round --save --loss_eval_freq 10 --num_consensus_rounds 10 --doubly_stoch metro
python main.py rand_walk round --loss_eval_freq 10 --graph_def wk_4 --numw 4

### Graph vs. distribution
python main.py --data_dist PQQQ --graph_def wk_4 --consensus rand_walk --num_consensus_rounds 5 --strag_dist equal --num_samples 64 --grad_combine Equal --weights_seed 99 --save --loss_eval_freq 10