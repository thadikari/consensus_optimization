#!/bin/bash

EXEC="python src/run_main.py --model toy --opt PG --consensus perfect --strag_dist bern --strag_dist_param 0.8 --num_samples 60 --grad_combine Equal Proportional --save --num_iters 10000"

log () { echo "$1"; }
run () { log ">> $1"; eval "$1" & }
exc () { run "$EXEC $1"; }


ARGS="--func linear2 --data_dist distinct_2_2 --toy_sigma2"
exc "$ARGS 0.001 --extra 0.001"
exc "$ARGS 0.01 --extra 0.01"
exc "$ARGS 0.1 --extra 0.1"
exc "$ARGS 10 --extra 10"
exc "$ARGS 100 --extra 100"

wait
