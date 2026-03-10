source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

TARGET_MODEL_PATH=""
PORT=8001

python3 -m sglang.launch_server \
  --model $TARGET_MODEL_PATH \
  --tp 8 \
  --dtype bfloat16 \
  --trust-remote-code \
  --attention-backend ascend \
  --device npu \
  --mem-fraction-static 0.65 \
  --log-level warning \
  --host 127.0.0.1 \
  --port $PORT \
  --max-running-requests 256 \
  --cuda-graph-max-bs 256
