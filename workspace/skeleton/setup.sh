#!/bin/bash

# Note that when we execute shell script file in bash,
# conda activate env && pip install ... not works properly.
# You should use conda run -n env <cmd>
# see the below example.
#
# If you want to create another environment, then set EVAL_ENV
# But we do not recommend it, because creating new environment
# and installing packages from scratch takes a lot of time,
# that can impact time limit of evaluation.
# 
# export EVAL_ENV=testenv
# conda create -n $EVAL_ENV python=3.10 -y
# conda run -n $EVAL_ENV pip install -r /workspace/evaluate/artifacts/requirements.txt
export EVAL_ENV=project
export HF_TOKEN="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"
conda run -n $EVAL_ENV pip install trl
conda run -n $EVAL_ENV pip install unsloth
conda run -n $EVAL_ENV pip install scikit-learn
conda run -n $EVAL_ENV pip install rich
conda run -n $EVAL_ENV pip install typing