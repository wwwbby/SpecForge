source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

TARGET_MODEL_PATH=""
DRAFT_MODEL_PATH=""
BATCH_SIZE=32
SPECULATIVE_NUM_STEPS=3
SPECULATIVE_EAGLE_TOPK=1
SPECULATIVE_NUM_DRAFT_TOKENS=4
PORT=8001

python3 -m sglang.launch_server \
  --model-path $TARGET_MODEL_PATH \
  --speculative-algorithm EAGLE3 \
  --disable-cuda-graph \
  --speculative-num-steps $SPECULATIVE_NUM_STEPS \
  --speculative-eagle-topk $SPECULATIVE_EAGLE_TOPK \
  --speculative-num-draft-tokens $SPECULATIVE_NUM_DRAFT_TOKENS \
  --speculative-draft-model-path $DRAFT_MODEL_PATH \
  --cuda-graph-max-bs 32 \
  --mem-fraction-static 0.8 \
  --tp-size 4 \
  --max-running-requests 32 \
  --trust-remote-code \
  --ep-size 1 \
  --attention-backend ascend \
  --dtype bfloat16 \
  --device npu \
  --host localhost \
  --port $PORT
