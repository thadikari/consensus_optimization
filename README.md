
# consensus

python main.py rand_walk round --save --loss_eval_freq 10 --num_consensus_rounds 10 --doubly_stoch metro
python main.py rand_walk round --loss_eval_freq 10 --graph_def wk_4 --numw 4

## graph vs distribution
python main.py --data_dist PQQQ --graph_def wk_4 --consensus rand_walk --num_consensus_rounds 5 --strag_dist equal --num_samples 64 --grad_combine Equal --weights_seed 99 --save --loss_eval_freq 10