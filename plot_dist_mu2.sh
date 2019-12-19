#!/bin/bash

EXEC="python -u src/plot_dist_mu2.py"
ARGS="--trials 100000 --noshow --save"

log () { echo "$1"; }
run () { log ">> $1"; eval "$1"; }
exc () { run "$EXEC $1 $ARGS"; }


exc "bern --bern_max 60 --bern_min 1"
exc "bern --bern_max 60 --bern_min 30"
exc "bern --bern_max 60 --bern_min 50"
exc "bern --bern_max 200 --bern_min 50"

exc "exp --exp_max 60 --exp_max_scale 15 --exp_sample_scale 10"
exc "exp --exp_max 200 --exp_max_scale 40 --exp_sample_scale 10"

exc "gauss --gauss_loc 60 --gauss_max_std 120 --ylog"
exc "gauss --gauss_loc 200 --gauss_max_std 400 --ylog"
