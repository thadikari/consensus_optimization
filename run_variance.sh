#!/bin/bash

EXEC="python src/run_main.py --dataset toy_model --opt PG --consensus perfect --grad_combine Equal Proportional --save --weights_seed 1 --eval_grad_var --num_iters 1 --strag_dist bern --strag_dist_param 0.8 --num_samples 60"

log () { echo "$1"; }
run () { log ">> $1"; eval "$1"; }   # eval "$1"; } if sequential eval "$1" & } if parallel
exc () { run "$EXEC $1"; }


ARGS1="--func linear2 --strategy toy_2_2 --toy_sigma2"
ARGS2="--func linear3 --strategy toy_4_3 --toy_sigma2"
for pwr in 0.001 0.01 0.1 1 10 100; do
    for mul in 1 2 4 6 8; do
        val=$(awk "BEGIN {printf \"%s\",${pwr}*${mul}}")
        exc "$ARGS2 $val --extra $val"
    done
done

wait
