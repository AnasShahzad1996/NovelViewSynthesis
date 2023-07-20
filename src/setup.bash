#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

source "${SCRIPT_DIR}/venv_old/bin/activate"

export LD_LIBRARY_PATH="/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-11.6/bin:$PATH"

export LD_LIBRARY_PATH="${SCRIPT_DIR}/venv_old/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
