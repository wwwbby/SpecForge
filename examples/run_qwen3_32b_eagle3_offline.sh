#!/bin/bash
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600
export PATH=/usr/local/Ascend/8.5.0/compiler/bishengir/bin:$PATH

export HCCL_BUFFSIZE=1024
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export STREAMS_PER_DEVICE=32
export INF_NAN_MODE_FORCE_DISABLE=1
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE="AIV"
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export ENABLE_ASCEND_MOE_NZ=1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

export SGLANG_PATH=""  # change to your own sglang path
export PYTHONPATH=$SGLANG_PATH:$PYTHONPATH

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=9003  # 可选：指定一个空闲端口
# 在启动脚本前设置环境变量
export GLOO_FORCE_IPV4=1
export NCCL_FORCE_IPV4=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export TRITON_LOG_LEVEL=WARNING
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NUM_GPUS=8
export TARGET_MODEL_PATH=""
export DRAFT_MODEL_CONFIG_RELATIVE_PATH=""
export DRAFT_MODEL_PATH=""
export TRAIN_DATA_PATH=""
export TRAIN_HIDDEN_STATES_PATH=""
export OUTPUT_DIR=""
export NUM_EPOCHS=2
export LEARNING_RATE=5e-5
export SAVE_INTERVAL=3000
export EVAL_INTERVAL=3000
export TTT_LENGTH=7
export EVAL_DATA_PATH=""
export EVAL_HIDDEN_STATES_PATH=""

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path $TARGET_MODEL_PATH \
    --draft-model-config $ROOT_DIR/$DRAFT_MODEL_CONFIG_RELATIVE_PATH \
    --train-data-path $TRAIN_DATA_PATH \
    --train-hidden-states-path $TRAIN_HIDDEN_STATES_PATH \
    --output-dir $OUTPUT_DIR \
    --num-epochs $NUM_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --max-length 2048 \
    --chat-template qwen \
    --target-model-backend sglang \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --ttt-length $TTT_LENGTH \
    --max-grad-norm 0.5 \
    --warmup-ratio 0.015 \
    --dist-timeout 60 \
    --eval-data-path $EVAL_DATA_PATH \
    --eval-hidden-states-path $EVAL_HIDDEN_STATES_PATH
