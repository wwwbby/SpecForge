source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

PROXY_FILE_PATH=""
source $PROXY_FILE_PATH
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

TARGET_MODEL_PATH=""
DRAFT_MODEL_PATH=""
BATCH_SIZE=32
SPECULATIVE_NUM_STEPS=3
SPECULATIVE_EAGLE_TOPK=1
SPECULATIVE_NUM_DRAFT_TOKENS=4
BENCHMARK_NAME="mtbench"
TEST_NUM_SAMPLES=80
PORT=8001

python benchmarks/bench_eagle3.py \
    --model-path $TARGET_MODEL_PATH \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path $DRAFT_MODEL_PATH \
    --config-list ${BATCH_SIZE},${SPECULATIVE_NUM_STEPS},${SPECULATIVE_EAGLE_TOPK},${SPECULATIVE_NUM_DRAFT_TOKENS} \
    --benchmark-list ${BENCHMARK_NAME}:${TEST_NUM_SAMPLES} \
    --dtype bfloat16 \
    --speculative-eagle-topk $SPECULATIVE_EAGLE_TOPK \
    --speculative-num-draft-tokens $SPECULATIVE_NUM_DRAFT_TOKENS \
    --speculative-num-steps $SPECULATIVE_NUM_STEPS \
    --tp 4 \
    --trust-remote-code \
    --attention-backend ascend \
    --device npu \
    --mem-fraction-static 0.65 \
    --log-level warning \
    --host 127.0.0.1 \
    --port $PORT \
    --max-running-requests 32 \
    --cuda-graph-max-bs 32 \
    --disable-cuda-graph \
    --output-dir ./results/
