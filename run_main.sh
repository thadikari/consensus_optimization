#!/bin/bash

EXEC="python -u src/run_main.py --model mnist --data_dist distinct_10 --save --graph_def amb_iclr_10 --num_iters 5000 --num_samples 60"

log () { echo "$1"; }
run () { log ">> $1"; eval "$1" & }
exc () { run "$EXEC $1"; }


#ARGS="--func linear1 --consensus perfect --strag_dist equal --grad_combine Equal --opt"
#exc "$ARGS PG"
#exc "$ARGS PWG"
#exc "$ARGS PWG1"
#exc "$ARGS PW"
#exc "$ARGS DA"

#exc "--opt PWG --consensus rand_walk --strag_dist equal --grad_combine Equal --num_consensus_rounds 1 --num_iters 100000"
#exc "--opt gd --consensus perfect --strag_dist bern --strag_dist_param 0.8 --grad_combine Equal Proportional"


ARGS="--opt PWG --strag_dist bern --strag_dist_param 0.8 --grad_combine Equal Proportional"
exc "$ARGS --func linear1 --consensus perfect"
exc "$ARGS --func linear1 --consensus rand_walk --num_consensus_rounds 10"
exc "$ARGS --func relu1 --consensus rand_walk --num_consensus_rounds 10"

wait
