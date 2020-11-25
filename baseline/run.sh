#!/bin/bash -e

DATADIR="datasets/"
KERNEL="tfidf"
PROBLEM="hate-offensive"

echo "Fitting..."
python3 classify.py \
    --mode train \
    --datadir $DATADIR \
    --kernel $KERNEL \
    --model-file "$KERNEL.${TOPICS%% *}.model" \
    --problem $PROBLEM

echo "Predicting..."
python3 classify.py \
    --mode test \
    --datadir $DATADIR \
    --model-file "$KERNEL.${TOPICS%% *}.model"  \
    --kernel $KERNEL \
    --predictions-file "$KERNEL.${TOPICS%% *}.predictions" \
    --problem $PROBLEM

echo "Computing Accuracy..."
python3 compute_accuracy.py \
    --datadir $DATADIR \
    --predictions-file "$KERNEL.${TOPICS%% *}.predictions" \
    --problem $PROBLEM
    